import os
import glob
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models # type: ignore
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# ==========================================
# Phase 1: Preprocessor — Faithful to Paper
# ==========================================
class TurbidityPreprocessor:
    """
    Faithfully implements Section 3.3 of the paper:
    a) Crop bottom (remove container feet)
    b) Enhance brightness
    c) Morphological ops (opening + closing)
    d) Filter contours by area (dot) or merge all (text)
    e) Tight bounding rect + 100px padding
    f) Normalise: mean = centre of dot area, variance = cropped rect std
    g) Two crops:
         - Dot:  upper edge crop  AND  lower edge crop (symmetric)
         - Text: top half ("The quick brown fox") AND bottom half ("jumps over the lazy dog")
    """

    # Paper uses 244x244 for dot datasets, 200x200 for text
    SIZE_MAP = {"dot": (244, 244), "text": (200, 200)}

    def __init__(self, pattern_type="dot"):
        self.pattern_type = pattern_type
        self.target_size = self.SIZE_MAP.get(pattern_type, (244, 244))

    def process(self, image_path):
        img = cv2.imread(image_path)
        if img is None:
            raise FileNotFoundError(f"Could not load {image_path}")

        h, w = img.shape[:2]

        if h > 800:
            # Full-size raw image: run the full pipeline
            img = self._crop_bottom(img)
            img = self._enhance_brightness(img)
            mask = self._morphological_ops(img)
            x, y, bw, bh = self._get_padded_bounding_box(mask, img.shape)
            roi = img[y:y+bh, x:x+bw]
            if roi.size == 0:
                roi = img
        else:
            # Already pre-cropped (244x244): use image directly
            roi = img

        roi_norm = self._normalize_image(roi)
        crop1, crop2 = self._final_crops(roi_norm)

        if crop1.size == 0 or crop2.size == 0:
            crop1 = roi_norm
            crop2 = roi_norm

        crop1 = cv2.resize(crop1, self.target_size)
        crop2 = cv2.resize(crop2, self.target_size)
        return crop1, crop2

    # ------------------------------------------------------------------
    def _crop_bottom(self, image, bottom_crop_pixels=600):
        h = image.shape[0]
        # Skip crop if image is already small (already pre-cropped)
        if h <= 800:
            return image
        return image[:max(1, h - bottom_crop_pixels), :]

    def _enhance_brightness(self, image, alpha=1.2, beta=30):
        return cv2.convertScaleAbs(image, alpha=alpha, beta=beta)

    def _morphological_ops(self, image):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        # Auto-detect: if image is mostly bright, dot is dark -> BINARY_INV
        # If image is mostly dark, dot is light -> BINARY
        mean_brightness = np.mean(gray)
        if mean_brightness > 127:
            _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        else:
            _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        kernel = np.ones((5, 5), np.uint8)
        opened  = cv2.morphologyEx(thresh, cv2.MORPH_OPEN,  kernel)
        closed  = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel)
        return closed

    def _get_padded_bounding_box(self, mask, img_shape):
        H, W = img_shape[:2]
        fallback = (W//4, H//4, W//2, H//2)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return fallback

        if self.pattern_type == "dot":
            # d) filter: keep largest contour (the dot)
            best = max(contours, key=cv2.contourArea)
            if cv2.contourArea(best) < 10:
                return fallback
            x, y, w, h = cv2.boundingRect(best)
        else:
            # d) text: merge all contours above noise threshold
            valid = [c for c in contours if cv2.contourArea(c) > 20]
            if not valid:
                return fallback
            xs = [cv2.boundingRect(c)[0] for c in valid]
            ys = [cv2.boundingRect(c)[1] for c in valid]
            x2s = [cv2.boundingRect(c)[0] + cv2.boundingRect(c)[2] for c in valid]
            y2s = [cv2.boundingRect(c)[1] + cv2.boundingRect(c)[3] for c in valid]
            x, y = min(xs), min(ys)
            w, h = max(x2s) - x, max(y2s) - y

        # e) 100px padding on all sides
        pad = 100
        px = max(0, x - pad);  py = max(0, y - pad)
        pw = min(W - px, w + 2*pad)
        ph = min(H - py, h + 2*pad)
        return px, py, pw, ph

    def _normalize_image(self, roi):
        """Simple [0,1] normalization. Z-score caused all-zero outputs."""
        return roi.astype(np.float32) / 255.0

    def _final_crops(self, norm_roi):
        """
        Paper Section 3.3g:
        Dot  → crop centred on UPPER EDGE of dot  +  crop centred on LOWER EDGE (symmetric)
        Text → top half  +  bottom half
        """
        h, w = norm_roi.shape[:2]

        if self.pattern_type == "dot":
            # The dot occupies roughly the full ROI height after padding.
            # Upper edge ≈ top 1/4 of roi, lower edge ≈ bottom 1/4 (symmetric).
            # We take a square centred on each edge.
            sq = min(w, h // 2)          # side length of each square crop
            half = sq // 2
            cx  = w // 2

            # Upper-edge crop: centred at (cx, h//4)
            uy = h // 4
            u_t = max(0, uy - half);  u_b = min(h, uy + half)
            u_l = max(0, cx - half);  u_r = min(w, cx + half)
            crop_upper = norm_roi[u_t:u_b, u_l:u_r]

            # Lower-edge crop: symmetric → centred at (cx, 3*h//4)
            ly = 3 * h // 4
            l_t = max(0, ly - half);  l_b = min(h, ly + half)
            crop_lower = norm_roi[l_t:l_b, u_l:u_r]

            return crop_upper, crop_lower

        else:   # text
            mid = h // 2
            return norm_roi[0:mid, :], norm_roi[mid:h, :]


# ==========================================
# Phase 2: Per-Dataset Splitting + Saving
# ==========================================
def split_and_process_individual_datasets(base_dir, output_dir, datasets):
    os.makedirs(output_dir, exist_ok=True)

    for ds_name, pattern in datasets.items():
        print(f"\n[{ds_name}] Starting processing...")
        ds_path = os.path.join(base_dir, ds_name)
        if not os.path.exists(ds_path):
            print(f"  [SKIP] Folder not found: {ds_path}")
            continue

        all_paths, all_labels = [], []
        for ntu_folder in os.listdir(ds_path):
            ntu_path = os.path.join(ds_path, ntu_folder)
            if not os.path.isdir(ntu_path):
                continue
            imgs = (glob.glob(os.path.join(ntu_path, "*.jpg")) +
                    glob.glob(os.path.join(ntu_path, "*.JPG")) +
                    glob.glob(os.path.join(ntu_path, "*.png")) +
                    glob.glob(os.path.join(ntu_path, "*.PNG")))
            for img in imgs:
                all_paths.append(img)
                all_labels.append(ntu_folder)

        if not all_paths:
            print(f"  [SKIP] No images found in {ds_path}")
            continue

        print(f"  Found {len(all_paths)} images across {len(set(all_labels))} classes: {sorted(set(all_labels), key=lambda x: float(x))}")

        # 3-way split: 15% test, 20% of remainder = val (~17% total), rest = train
        X_temp, X_test, y_temp, y_test = train_test_split(
            all_paths, all_labels, test_size=0.15, random_state=42, stratify=all_labels)
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp, test_size=0.20, random_state=42, stratify=y_temp)

        processor = TurbidityPreprocessor(pattern_type=pattern)

        for split_name, paths, labels in [("train", X_train, y_train),
                                           ("val",   X_val,   y_val),
                                           ("test",  X_test,  y_test)]:
            ok, fail = 0, 0
            for path, label in zip(paths, labels):
                out_dir = os.path.join(output_dir, ds_name, split_name, label)
                os.makedirs(out_dir, exist_ok=True)
                base = os.path.splitext(os.path.basename(path))[0]
                try:
                    c1, c2 = processor.process(path)
                    np.save(os.path.join(out_dir, f"{base}_c1.npy"), c1)
                    np.save(os.path.join(out_dir, f"{base}_c2.npy"), c2)
                    ok += 1
                except Exception as e:
                    fail += 1
            print(f"  {split_name:5s}: {ok} saved, {fail} failed")

    print("\nDone processing all datasets.")


# ==========================================
# Phase 3: Data Generator
# ==========================================
class NpyDataGenerator(tf.keras.utils.Sequence):
    def __init__(self, file_paths, labels, batch_size=32,
                 shuffle=True, augment=False):
        self.file_paths = np.array(file_paths)
        self.labels     = np.array(labels)
        self.batch_size = batch_size
        self.shuffle    = shuffle
        self.augment    = augment
        self.indices    = np.arange(len(self.file_paths))
        if self.shuffle:
            np.random.shuffle(self.indices)

    def __len__(self):
        return int(np.ceil(len(self.file_paths) / self.batch_size))

    def __getitem__(self, idx):
        bi = self.indices[idx * self.batch_size:(idx + 1) * self.batch_size]
        X  = np.array([np.load(str(self.file_paths[i])) for i in bi])
        y  = self.labels[bi]
        if self.augment:
            X = self._augment(X)
        return X, y

    def _augment(self, X):
        out = []
        for img in X:
            # Vertical flip: dot can be top or bottom
            if np.random.rand() > 0.5:
                img = np.flipud(img)
            # Horizontal flip: symmetric pattern
            if np.random.rand() > 0.5:
                img = np.fliplr(img)
            # Mild brightness jitter ±10%
            img = np.clip(img * (1.0 + np.random.uniform(-0.1, 0.1)), 0, 1)
            out.append(img)
        return np.array(out)

    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.indices)


def get_paths_and_labels(folder_path):
    paths, labels = [], []
    if not os.path.exists(folder_path):
        return np.array(paths), np.array(labels)
    for root, _, files in os.walk(folder_path):
        for f in files:
            if f.endswith(".npy"):
                paths.append(os.path.join(root, f))
                labels.append(os.path.basename(root))
    return np.array(paths), np.array(labels)


# ==========================================
# Phase 4: Paper's Exact CNN Architecture
# ==========================================
def build_paper_cnn(input_shape, num_classes):
    """
    Section 3.4 of the paper:
    - 5 conv layers: depths 16, 32, 64, 128, 256
    - kernel 3x3, ReLU
    - AveragePooling2D between every conv layer   ← NOT MaxPooling
    - Classification head: Dense(num_classes) + Softmax
    - AveragePooling preserves the soft blur gradient that IS the turbidity signal
    """
    inp = layers.Input(shape=input_shape)

    x = layers.Conv2D(16,  (3,3), activation='relu', padding='same')(inp)
    x = layers.AveragePooling2D((2,2))(x)

    x = layers.Conv2D(32,  (3,3), activation='relu', padding='same')(x)
    x = layers.AveragePooling2D((2,2))(x)

    x = layers.Conv2D(64,  (3,3), activation='relu', padding='same')(x)
    x = layers.AveragePooling2D((2,2))(x)

    x = layers.Conv2D(128, (3,3), activation='relu', padding='same')(x)
    x = layers.AveragePooling2D((2,2))(x)

    x = layers.Conv2D(256, (3,3), activation='relu', padding='same')(x)
    x = layers.AveragePooling2D((2,2))(x)

    x = layers.Flatten()(x)
    x = layers.Dropout(0.3)(x)
    out = layers.Dense(num_classes, activation='softmax')(x)

    model = models.Model(inp, out)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001, decay=0.001),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    return model


# ==========================================
# Phase 5: Main
# ==========================================
if __name__ == "__main__":
    BASE_DIR      = "."
    PROCESSED_DIR = "./PROCESSED_DATA_PAPER"

    # Match paper's dataset names exactly — adjust these to your actual folder names
    DATASETS = {
        "Clear Plastic": "dot",
        "Formazine":     "dot",
        "Kaolin":        "dot",
        "Text":          "text",
    }

    # ── Step 1: Process raw images ──────────────────────────────────────────
    # Uncomment on first run (or when raw data changes):
    split_and_process_individual_datasets(BASE_DIR, PROCESSED_DIR, DATASETS)

    # ── Step 2: Train one model per dataset ─────────────────────────────────
    for ds_name, pattern in DATASETS.items():
        print("\n" + "="*60)
        print(f"  PIPELINE: {ds_name.upper()}")
        print("="*60)

        ds_dir = os.path.join(PROCESSED_DIR, ds_name)
        if not os.path.exists(ds_dir):
            print(f"  [SKIP] Processed dir not found: {ds_dir}")
            continue

        train_paths, train_labels_str = get_paths_and_labels(os.path.join(ds_dir, "train"))
        val_paths,   val_labels_str   = get_paths_and_labels(os.path.join(ds_dir, "val"))
        test_paths,  test_labels_str  = get_paths_and_labels(os.path.join(ds_dir, "test"))

        # Diagnostics: catch empty-split issues early
        for split, p in [("train", train_paths), ("val", val_paths), ("test", test_paths)]:
            print(f"  {split}: {len(p)} samples")
            if len(p) == 0:
                print(f"  [WARN] {split} split is EMPTY — check folder names in {ds_dir}")

        if len(train_paths) == 0:
            print("  [SKIP] No training data.")
            continue

        # Encode labels in sorted numeric order (critical for ordinal meaning)
        all_str = np.concatenate([train_labels_str, val_labels_str, test_labels_str])
        unique_sorted = sorted(set(all_str), key=lambda x: float(x))
        encoder = LabelEncoder()
        encoder.fit(unique_sorted)

        y_train = encoder.transform(train_labels_str)
        y_val   = encoder.transform(val_labels_str)
        y_test  = encoder.transform(test_labels_str)

        print(f"  Classes ({len(unique_sorted)}): {unique_sorted}")

        # Paper uses batch_size=32, augment only training
        train_gen = NpyDataGenerator(train_paths, y_train, batch_size=32,
                                     shuffle=True,  augment=True)
        val_gen   = NpyDataGenerator(val_paths,   y_val,   batch_size=32,
                                     shuffle=False, augment=False)
        test_gen  = NpyDataGenerator(test_paths,  y_test,  batch_size=32,
                                     shuffle=False, augment=False)

        # Input shape matches paper: 244x244 for dot, 200x200 for text
        target_size = TurbidityPreprocessor.SIZE_MAP.get(pattern, (244, 244))
        input_shape = (*target_size, 3)

        model = build_paper_cnn(input_shape, len(unique_sorted))
        model.summary()

        callbacks = [
            tf.keras.callbacks.EarlyStopping(
                monitor='val_accuracy', patience=8,
                restore_best_weights=True, verbose=1),
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss', factor=0.5,
                patience=3, min_lr=1e-6, verbose=1),
        ]

        # Paper trains for 10 epochs (formazine) but we allow up to 50 with early stop
        print(f"\n  Training {ds_name}...")
        model.fit(train_gen, epochs=40, validation_data=val_gen, callbacks=callbacks)

        # Save
        safe = ds_name.replace(' ', '_')
        model.save(f"model_{safe}_paper.keras")
        np.save(f"classes_{safe}_paper.npy", encoder.classes_)
        print(f"  Saved model_{safe}_paper.keras")

        # Evaluate
        print(f"\n  Final test evaluation: {ds_name}")
        _, test_acc = model.evaluate(test_gen, verbose=1)
        print(f"  >>> {ds_name} Exact Accuracy:        {test_acc*100:.2f}% <<<")

        # Within-1-class accuracy (useful for near-misses on ordered NTU scale)
        all_pred, all_true = [], []
        for i in range(len(test_gen)):
            xb, yb = test_gen[i]
            preds = np.argmax(model.predict(xb, verbose=0), axis=-1)
            all_pred.extend(preds); all_true.extend(yb)
        all_pred = np.array(all_pred); all_true = np.array(all_true)
        w1 = np.mean(np.abs(all_pred - all_true) <= 1)
        print(f"  >>> {ds_name} Within-1-Class Accuracy: {w1*100:.2f}% <<<")

        tf.keras.backend.clear_session()