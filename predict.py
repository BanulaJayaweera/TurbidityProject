import os
import sys
import argparse
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import tensorflow as tf

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from turbidity_match import TurbidityPreprocessor

DATASET_CONFIG = {
    "Clear_Plastic": "dot",
    "Formazine":     "dot",
    "Kaolin":        "dot",
    "Text":          "text",
}


def load_model_and_classes(dataset):
    model_path   = f"model_{dataset}.keras"
    classes_path = f"classes_{dataset}.npy"
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found: {model_path}")
    if not os.path.exists(classes_path):
        raise FileNotFoundError(f"Classes file not found: {classes_path}")
    model   = tf.keras.models.load_model(model_path)
    classes = np.load(classes_path, allow_pickle=True)
    return model, classes


def get_stages(image_path, pattern_type):
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Could not load image: {image_path}")

    processor = TurbidityPreprocessor(pattern_type=pattern_type)
    stages = {"Original": cv2.cvtColor(img, cv2.COLOR_BGR2RGB)}

    h = img.shape[0]
    if h > 800:
        cropped = processor._crop_bottom(img)
        stages["Bottom Cropped"] = cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB)

        brightened = processor._enhance_brightness(cropped)
        stages["Brightness Enhanced"] = cv2.cvtColor(brightened, cv2.COLOR_BGR2RGB)

        mask = processor._morphological_ops(brightened)
        stages["Morphological Mask"] = mask

        x, y, bw, bh = processor._get_padded_bounding_box(mask, brightened.shape)
        bbox_img = brightened.copy()
        cv2.rectangle(bbox_img, (x, y), (x + bw, y + bh), (0, 255, 0), 4)
        stages["Bounding Box"] = cv2.cvtColor(bbox_img, cv2.COLOR_BGR2RGB)

        roi = brightened[y:y + bh, x:x + bw]
        if roi.size == 0:
            roi = brightened
        stages["ROI Crop"] = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
    else:
        roi = img
        stages["Image (pre-cropped)"] = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    roi_norm = processor._normalize_image(roi)
    stages["Normalized ROI"] = roi_norm

    crop1, crop2 = processor._final_crops(roi_norm)
    if crop1.size == 0 or crop2.size == 0:
        crop1 = crop2 = roi_norm

    crop1 = cv2.resize(crop1, processor.target_size)
    crop2 = cv2.resize(crop2, processor.target_size)

    label1 = "Crop 1 (Upper)" if pattern_type == "dot" else "Crop 1 (Top half)"
    label2 = "Crop 2 (Lower)" if pattern_type == "dot" else "Crop 2 (Bottom half)"
    stages[label1] = crop1
    stages[label2] = crop2

    return stages, crop1, crop2


def predict(model, classes, crop1, crop2):
    prob1 = model.predict(crop1[np.newaxis], verbose=0)[0]
    prob2 = model.predict(crop2[np.newaxis], verbose=0)[0]
    avg_prob  = (prob1 + prob2) / 2
    pred_idx  = np.argmax(avg_prob)
    return classes[pred_idx], avg_prob, pred_idx


def visualize(stages, predicted_ntu, actual_ntu, dataset, avg_prob, classes):
    n_stages = len(stages)
    fig = plt.figure(figsize=(max(18, n_stages * 3), 10))
    gs  = gridspec.GridSpec(2, n_stages, figure=fig, hspace=0.45, wspace=0.3)

    for i, (name, img) in enumerate(stages.items()):
        ax = fig.add_subplot(gs[0, i])
        if img.ndim == 2:
            ax.imshow(img, cmap="gray")
        else:
            ax.imshow(np.clip(img, 0, 1) if img.dtype == np.float32 else img)
        ax.set_title(name, fontsize=9, pad=4)
        ax.axis("off")

    # Probability bar chart
    ax_bar = fig.add_subplot(gs[1, :])
    pred_idx   = np.argmax(avg_prob)
    actual_idx = None
    if actual_ntu is not None:
        matches = [str(c) == str(actual_ntu) for c in classes]
        if any(matches):
            actual_idx = matches.index(True)

    colors = ["#2196F3"] * len(classes)
    if actual_idx is not None:
        colors[actual_idx] = "#4CAF50"        # green  = actual
    colors[pred_idx] = "#F44336"              # red    = predicted
    if actual_idx is not None and actual_idx == pred_idx:
        colors[pred_idx] = "#FF9800"          # orange = correct prediction

    bars = ax_bar.bar(range(len(classes)), avg_prob * 100, color=colors)
    ax_bar.set_xticks(range(len(classes)))
    ax_bar.set_xticklabels([str(c) for c in classes], rotation=45, ha="right")
    ax_bar.set_ylabel("Confidence (%)")
    ax_bar.set_xlabel("NTU Class")
    ax_bar.set_ylim(0, 105)
    for bar, prob in zip(bars, avg_prob):
        if prob > 0.02:
            ax_bar.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                        f"{prob*100:.1f}%", ha="center", va="bottom", fontsize=7)

    legend_patches = [
        plt.Rectangle((0, 0), 1, 1, color="#F44336", label="Predicted"),
        plt.Rectangle((0, 0), 1, 1, color="#4CAF50", label="Actual"),
        plt.Rectangle((0, 0), 1, 1, color="#FF9800", label="Correct (predicted = actual)"),
        plt.Rectangle((0, 0), 1, 1, color="#2196F3", label="Other"),
    ]
    ax_bar.legend(handles=legend_patches, loc="upper right", fontsize=8)

    correct = actual_ntu is not None and str(predicted_ntu) == str(actual_ntu)
    if actual_ntu is not None:
        result_str = "CORRECT" if correct else f"WRONG  (actual: {actual_ntu} NTU)"
        title_color = "green" if correct else "red"
    else:
        result_str = ""
        title_color = "black"

    title = (f"Dataset: {dataset}  |  Predicted: {predicted_ntu} NTU  |  "
             f"Confidence: {avg_prob[pred_idx]*100:.1f}%"
             + (f"  |  {result_str}" if result_str else ""))
    fig.suptitle(title, fontsize=13, fontweight="bold", color=title_color)

    plt.tight_layout()
    plt.show()


def main():
    parser = argparse.ArgumentParser(
        description="Predict turbidity NTU with preprocessing visualization"
    )
    parser.add_argument("image",   help="Path to the input image")
    parser.add_argument("dataset", choices=list(DATASET_CONFIG.keys()),
                        help="Dataset/model to use: Clear_Plastic, Formazine, Kaolin, Text")
    parser.add_argument("--actual", default=None,
                        help="Actual NTU value for accuracy check (e.g. --actual 25)")
    args = parser.parse_args()

    # Auto-detect actual NTU from parent folder name if not provided
    if args.actual is None:
        parent = os.path.basename(os.path.dirname(os.path.abspath(args.image)))
        try:
            float(parent)
            args.actual = parent
            print(f"  Auto-detected actual NTU from folder name: {parent}")
        except ValueError:
            pass

    pattern = DATASET_CONFIG[args.dataset]

    print(f"Loading model: model_{args.dataset}.keras ...")
    model, classes = load_model_and_classes(args.dataset)

    print("Preprocessing image...")
    stages, crop1, crop2 = get_stages(args.image, pattern)

    print("Running prediction...")
    predicted_ntu, avg_prob, pred_idx = predict(model, classes, crop1, crop2)

    print(f"\n  Predicted NTU : {predicted_ntu}  (confidence: {avg_prob[pred_idx]*100:.1f}%)")
    if args.actual:
        match = str(predicted_ntu) == str(args.actual)
        print(f"  Actual NTU    : {args.actual}")
        print(f"  Result        : {'CORRECT' if match else 'WRONG'}")

    visualize(stages, predicted_ntu, args.actual, args.dataset, avg_prob, classes)


if __name__ == "__main__":
    main()
