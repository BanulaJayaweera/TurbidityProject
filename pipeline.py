import os
import glob
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models # type: ignore
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# ==========================================
# Phase 1: FIXED Preprocessor
# ==========================================
class TurbidityPreprocessor:
    def __init__(self, pattern_type="dot", target_size=(128, 128)):
        self.pattern_type = pattern_type
        self.target_size = target_size

    def process(self, image_path):
        img = cv2.imread(image_path)
        if img is None:
            raise FileNotFoundError(f"Could not load {image_path}")

        img_cropped = self._crop_bottom(img)
        img_bright = self._enhance_brightness(img_cropped)

        if self.pattern_type == "dot":
            return self._process_dot(img_bright)
        else:
            return self._process_text(img_bright)

    def _process_dot(self, img):
        """
        FIX: Instead of hardcoded top/bottom crops, we:
        1. Auto-detect whether dot is dark-on-light OR light-on-dark
        2. Find the actual dot bounding box using the correct threshold direction
        3. Return ONE centered crop around the dot (not two positional guesses)
        We return (crop, crop) so the downstream code still receives a tuple,
        but both values are the same valid crop — no garbage crops.
        """
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # --- Step 1: Auto-detect inversion ---
        # The dot is large. If the image mean is dark, dot is dark-on-light (normal).
        # If the image mean is bright with a dark blob, same. We compare
        # the overall mean vs the corner regions (which are typically background).
        h, w = gray.shape
        corner_size = min(h, w) // 6
        corners = [
            gray[:corner_size, :corner_size],
            gray[:corner_size, -corner_size:],
            gray[-corner_size:, :corner_size],
            gray[-corner_size:, -corner_size:]
        ]
        bg_mean = np.mean([c.mean() for c in corners])
        center_mean = gray[h//4:3*h//4, w//4:3*w//4].mean()

        # If background (corners) is brighter than center → dot is dark on light → BINARY_INV
        # If background is darker than center → dot is light on dark → BINARY (no inv)
        if bg_mean > center_mean:
            _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        else:
            _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # Clean up noise
        kernel = np.ones((7, 7), np.uint8)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

        # --- Step 2: Find the largest contour (the dot) ---
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if not contours:
            # Fallback: just resize the whole image
            fallback = cv2.resize(img, self.target_size).astype(np.float32) / 255.0
            return fallback, fallback

        best_cnt = max(contours, key=cv2.contourArea)
        if cv2.contourArea(best_cnt) < 50:
            fallback = cv2.resize(img, self.target_size).astype(np.float32) / 255.0
            return fallback, fallback

        # --- Step 3: Crop around the detected dot (with padding) ---
        x, y, bw, bh = cv2.boundingRect(best_cnt)
        pad = 40
        x1 = max(0, x - pad)
        y1 = max(0, y - pad)
        x2 = min(w, x + bw + pad)
        y2 = min(h, y + bh + pad)

        dot_crop = img[y1:y2, x1:x2]

        if dot_crop.size == 0:
            fallback = cv2.resize(img, self.target_size).astype(np.float32) / 255.0
            return fallback, fallback

        # --- Step 4: Normalize to [0, 1] (simple, robust) ---
        crop_resized = cv2.resize(dot_crop, self.target_size).astype(np.float32) / 255.0

        # Return same crop twice to keep tuple interface
        return crop_resized, crop_resized

    def _process_text(self, img):
        """Text pattern: split image in half horizontally as before."""
        h, w = img.shape[:2]
        norm = img.astype(np.float32) / 255.0
        mid = h // 2
        crop1 = cv2.resize(norm[0:mid, :], self.target_size)
        crop2 = cv2.resize(norm[mid:h, :], self.target_size)
        return crop1, crop2

    def _crop_bottom(self, image, bottom_crop_pixels=600):
        h = image.shape[0]
        return image[:max(1, h - bottom_crop_pixels), :]

    def _enhance_brightness(self, image, alpha=1.2, beta=30):
        return cv2.convertScaleAbs(image, alpha=alpha, beta=beta)


# ==========================================
# Phase 2: Per-Dataset Splitting
# ==========================================
def split_and_process_individual_datasets(base_dir, output_dir, datasets):
    os.makedirs(output_dir, exist_ok=True)

    for ds_name, pattern in datasets.items():
        print(f"\n[{ds_name}] Starting processing...")
        ds_path = os.path.join(base_dir, ds_name)
        if not os.path.exists(ds_path):
            print(f"[{ds_name}] Folder not found, skipping.")
            continue

        all_paths, all_labels = [], []

        for ntu_folder in os.listdir(ds_path):
            ntu_path = os.path.join(ds_path, ntu_folder)
            if not os.path.isdir(ntu_path):
                continue
            images = glob.glob(os.path.join(ntu_path, "*.jpg")) + \
                     glob.glob(os.path.join(ntu_path, "*.png"))
            for img in images:
                all_paths.append(img)
                all_labels.append(ntu_folder)

        if not all_paths:
            print(f"[{ds_name}] No images found, skipping.")
            continue

        X_temp, X_test, y_temp, y_test = train_test_split(
            all_paths, all_labels, test_size=0.15, random_state=42, stratify=all_labels
        )
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp, test_size=0.20, random_state=42, stratify=y_temp
        )

        splits = [("train", X_train, y_train), ("val", X_val, y_val), ("test", X_test, y_test)]
        processor = TurbidityPreprocessor(pattern_type=pattern, target_size=(128, 128))

        for split_name, paths, labels in splits:
            print(f"  -> Processing {split_name} set ({len(paths)} images)...")
            for i, path in enumerate(paths):
                label = labels[i]
                out_dir = os.path.join(output_dir, ds_name, split_name, label)
                os.makedirs(out_dir, exist_ok=True)

                base_name = os.path.splitext(os.path.basename(path))[0]
                try:
                    c1, c2 = processor.process(path)
                    np.save(os.path.join(out_dir, f"{base_name}_c1.npy"), c1)

                    # For dot patterns both crops are identical, only save c2 for text
                    if pattern == "text":
                        np.save(os.path.join(out_dir, f"{base_name}_c2.npy"), c2)
                except Exception as e:
                    print(f"    [WARN] Failed on {path}: {e}")

    print("\nAll datasets split and processed successfully!")


# ==========================================
# Phase 3: Generators & CNN
# ==========================================
class NpyDataGenerator(tf.keras.utils.Sequence):
    def __init__(self, file_paths, labels, batch_size=32, shuffle=True, augment=False):
        self.file_paths = file_paths
        self.labels = labels
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.augment = augment
        self.indices = np.arange(len(self.file_paths))
        if self.shuffle:
            np.random.shuffle(self.indices)

    def __len__(self):
        return int(np.ceil(len(self.file_paths) / self.batch_size))

    def __getitem__(self, index):
        batch_indices = self.indices[index * self.batch_size:(index + 1) * self.batch_size]
        X = np.array([np.load(self.file_paths[i]) for i in batch_indices])
        y = np.array([self.labels[i] for i in batch_indices])

        if self.augment:
            X = self._augment_batch(X)

        return X, y

    def _augment_batch(self, X):
        """
        Simple augmentation: random horizontal/vertical flips + brightness jitter.
        Since the dot can appear top or bottom, vertical flip is valid augmentation.
        """
        out = []
        for img in X:
            if np.random.rand() > 0.5:
                img = np.fliplr(img)
            if np.random.rand() > 0.5:
                img = np.flipud(img)
            # Brightness jitter ±10%
            factor = 1.0 + np.random.uniform(-0.1, 0.1)
            img = np.clip(img * factor, 0.0, 1.0)
            out.append(img)
        return np.array(out)

    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.indices)


def get_paths_and_labels(folder_path):
    paths, labels = [], []
    for root, _, files in os.walk(folder_path):
        for file in files:
            if file.endswith(".npy"):
                paths.append(os.path.join(root, file))
                labels.append(os.path.basename(root))
    return np.array(paths), np.array(labels)


# ==========================================
# Ordinal Loss: treats NTU classes as ordered
# ==========================================
def ordinal_loss(y_true, y_pred):
    """
    Mean-Absolute-Error in label space.
    Penalises predicting 0 NTU when the truth is 1000 NTU much more than
    a one-step error. Works with integer class indices that map to ordered NTU values.
    Used ALONGSIDE cross-entropy (combined loss below).
    """
    y_true_f = tf.cast(y_true, tf.float32)
    pred_class = tf.cast(tf.argmax(y_pred, axis=-1), tf.float32)
    return tf.reduce_mean(tf.abs(y_true_f - pred_class))


def combined_loss(num_classes):
    """Weighted sum of cross-entropy + ordinal MAE."""
    cce = tf.keras.losses.SparseCategoricalCrossentropy()

    def loss_fn(y_true, y_pred):
        ce = cce(y_true, y_pred)
        ordinal = ordinal_loss(y_true, y_pred)
        # Normalise ordinal term so it's on a similar scale to CE
        ordinal_norm = ordinal / tf.cast(num_classes, tf.float32)
        return ce + 0.5 * ordinal_norm

    return loss_fn


# ==========================================
# Phase 4: Improved CNN with BatchNorm + Deeper
# ==========================================
def build_classification_cnn(input_shape, num_classes):
    """
    Deeper CNN with BatchNorm after each conv block.
    BatchNorm helps a lot when input normalisation is imperfect
    (which it will be given the variable dot positions and lighting).
    """
    inputs = layers.Input(shape=input_shape)

    x = layers.Conv2D(32, (3, 3), padding='same')(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.MaxPooling2D((2, 2))(x)

    x = layers.Conv2D(64, (3, 3), padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.MaxPooling2D((2, 2))(x)

    x = layers.Conv2D(128, (3, 3), padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.MaxPooling2D((2, 2))(x)

    x = layers.Conv2D(128, (3, 3), padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.GlobalAveragePooling2D()(x)   # Better than Flatten for small datasets

    x = layers.Dense(256, activation='relu')(x)
    x = layers.Dropout(0.4)(x)
    x = layers.Dense(128, activation='relu')(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(num_classes, activation='softmax')(x)

    model = models.Model(inputs, outputs)
    return model


# ==========================================
# Phase 5: Main Execution
# ==========================================
if __name__ == "__main__":
    BASE_DIR = "."
    PROCESSED_DIR = "./PROCESSED_DATA_4"
    DATASETS = {
        "Clear Plastic": "dot",
        "Formazine": "dot",
        "Kaolin": "dot",
        "Text": "text"
    }

    # --- 1. RUN PROCESSING ---
    # Uncomment when running for the first time:
    split_and_process_individual_datasets(BASE_DIR, PROCESSED_DIR, DATASETS)

    # --- 2. TRAIN 4 SEPARATE MODELS ---
    for ds_name, pattern in DATASETS.items():
        print("\n" + "=" * 60)
        print(f"  STARTING PIPELINE FOR: {ds_name.upper()}")
        print("=" * 60)

        ds_processed_dir = os.path.join(PROCESSED_DIR, ds_name)
        if not os.path.exists(ds_processed_dir):
            print(f"Processed data for {ds_name} not found. Skipping.")
            continue

        train_paths, train_labels_str = get_paths_and_labels(
            os.path.join(ds_processed_dir, "train"))
        val_paths, val_labels_str = get_paths_and_labels(
            os.path.join(ds_processed_dir, "val"))
        test_paths, test_labels_str = get_paths_and_labels(
            os.path.join(ds_processed_dir, "test"))

        if len(train_paths) == 0:
            print(f"No training data found for {ds_name}. Skipping.")
            continue

        # Sort classes numerically (important for ordinal loss to be meaningful)
        all_str_labels = np.concatenate([train_labels_str, val_labels_str, test_labels_str])
        unique_labels = sorted(set(all_str_labels), key=lambda x: float(x))
        encoder = LabelEncoder()
        encoder.fit(unique_labels)   # Fit in sorted numeric order

        y_train = encoder.transform(train_labels_str)
        y_val = encoder.transform(val_labels_str)
        y_test = encoder.transform(test_labels_str)

        print(f"Classes (sorted): {encoder.classes_}")
        print(f"Data -> Train: {len(train_paths)} | Val: {len(val_paths)} | Test: {len(test_paths)}")

        # Augment only training data
        train_gen = NpyDataGenerator(train_paths, y_train, batch_size=32,
                                     shuffle=True, augment=True)
        val_gen   = NpyDataGenerator(val_paths,   y_val,   batch_size=32,
                                     shuffle=False, augment=False)
        test_gen  = NpyDataGenerator(test_paths,  y_test,  batch_size=32,
                                     shuffle=False, augment=False)

        num_classes = len(encoder.classes_)
        model = build_classification_cnn((128, 128, 3), num_classes)

        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
            loss=combined_loss(num_classes),
            metrics=['accuracy']
        )

        model.summary()

        callbacks = [
            tf.keras.callbacks.EarlyStopping(
                monitor='val_accuracy',
                patience=8,
                restore_best_weights=True,
                verbose=1
            ),
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=3,
                min_lr=1e-6,
                verbose=1
            )
        ]

        print(f"\n--- Training {ds_name} Model ---")
        model.fit(
            train_gen,
            epochs=50,
            validation_data=val_gen,
            callbacks=callbacks
        )

        model_filename = f"model_{ds_name.replace(' ', '_')}_v2.keras"
        class_filename = f"classes_{ds_name.replace(' ', '_')}_v2.npy"
        model.save(model_filename)
        np.save(class_filename, encoder.classes_)
        print(f"Saved: '{model_filename}'")

        print(f"\n--- Final Test Evaluation: {ds_name} ---")
        test_loss, test_acc = model.evaluate(test_gen)
        print(f">>> {ds_name} Accuracy: {test_acc * 100:.2f}% <<<")

        # Also report within-1-class accuracy (useful for ordinal problems)
        all_preds, all_true = [], []
        for i in range(len(test_gen)):
            xb, yb = test_gen[i]
            preds = np.argmax(model.predict(xb, verbose=0), axis=-1)
            all_preds.extend(preds)
            all_true.extend(yb)

        all_preds = np.array(all_preds)
        all_true = np.array(all_true)
        within_1 = np.mean(np.abs(all_preds - all_true) <= 1)
        print(f">>> {ds_name} Within-1-Class Accuracy: {within_1 * 100:.2f}% <<<")

        tf.keras.backend.clear_session()