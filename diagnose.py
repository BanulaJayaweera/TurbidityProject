"""
DIAGNOSTIC SCRIPT — run this before retraining.
It checks the actual processed .npy files to find why the model isn't learning.
"""
import os
import numpy as np
import cv2
import glob
from collections import Counter

PROCESSED_DIR = "./PROCESSED_DATA_PAPER"
DS_NAME       = "Formazine"   # Start with the largest dataset

# ============================================================
# Check 1: Class distribution
# ============================================================
print("=" * 60)
print("CHECK 1: Class distribution in train split")
print("=" * 60)
train_dir = os.path.join(PROCESSED_DIR, DS_NAME, "train")
if not os.path.exists(train_dir):
    print(f"ERROR: {train_dir} does not exist!")
else:
    class_counts = {}
    for cls in os.listdir(train_dir):
        cls_path = os.path.join(train_dir, cls)
        if os.path.isdir(cls_path):
            n = len(glob.glob(os.path.join(cls_path, "*.npy")))
            class_counts[cls] = n
    sorted_classes = sorted(class_counts.items(), key=lambda x: float(x[0]))
    print(f"{'Class (NTU)':<15} {'Count':<10}")
    print("-" * 25)
    total = 0
    for cls, cnt in sorted_classes:
        print(f"{cls:<15} {cnt:<10}")
        total += cnt
    print(f"{'TOTAL':<15} {total:<10}")
    print(f"\nRandom-chance accuracy would be: {1/len(sorted_classes)*100:.1f}%")

# ============================================================
# Check 2: Inspect actual pixel values of saved .npy files
# ============================================================
print("\n" + "=" * 60)
print("CHECK 2: Pixel value ranges in saved .npy files")
print("=" * 60)
sample_files = []
if os.path.exists(train_dir):
    for cls in os.listdir(train_dir):
        cls_path = os.path.join(train_dir, cls)
        if os.path.isdir(cls_path):
            files = glob.glob(os.path.join(cls_path, "*.npy"))[:2]
            sample_files.extend(files)

issues_found = []
for f in sample_files[:10]:
    arr = np.load(f)
    has_nan = np.any(np.isnan(arr))
    has_inf = np.any(np.isinf(arr))
    label   = os.path.basename(os.path.dirname(f))
    print(f"  [{label}] shape={arr.shape} min={arr.min():.3f} "
          f"max={arr.max():.3f} mean={arr.mean():.3f} "
          f"NaN={has_nan} Inf={has_inf}")
    if has_nan or has_inf:
        issues_found.append(f"NaN/Inf in {f}")
    if arr.max() > 100 or arr.min() < -100:
        issues_found.append(f"Extreme values in {f}: [{arr.min():.1f}, {arr.max():.1f}]")

if issues_found:
    print("\n!! ISSUES FOUND !!")
    for i in issues_found:
        print(f"  {i}")
else:
    print("\n  Values look OK.")

# ============================================================
# Check 3: Visualise a few crops per class
# ============================================================
print("\n" + "=" * 60)
print("CHECK 3: Saving sample crops as images for visual inspection")
print("=" * 60)
os.makedirs("./debug_crops", exist_ok=True)

if os.path.exists(train_dir):
    for cls in sorted(os.listdir(train_dir), key=lambda x: float(x)):
        cls_path = os.path.join(train_dir, cls)
        if not os.path.isdir(cls_path):
            continue
        files = glob.glob(os.path.join(cls_path, "*.npy"))[:3]
        for i, f in enumerate(files):
            arr = np.load(f)
            # Convert normalised float back to uint8 for viewing
            # Handle both [0,1] range and z-score normalised range
            a_min, a_max = arr.min(), arr.max()
            if a_max - a_min < 1e-6:
                vis = np.zeros_like(arr, dtype=np.uint8)
            else:
                vis = ((arr - a_min) / (a_max - a_min) * 255).astype(np.uint8)
            out_path = f"./debug_crops/{DS_NAME}_class{cls}_sample{i}.png"
            cv2.imwrite(out_path, vis)

    print(f"  Saved crops to ./debug_crops/")
    print(f"  Open these images to see what the model is actually receiving.")
    print(f"  If they look like noise/blank/wrong → preprocessing is broken.")
    print(f"  If they clearly show the dot edge → model architecture is the problem.")

# ============================================================
# Check 4: Are c1 and c2 identical? (dot datasets)
# ============================================================
print("\n" + "=" * 60)
print("CHECK 4: Are c1 and c2 crops different from each other?")
print("=" * 60)
if os.path.exists(train_dir):
    cls = os.listdir(train_dir)[0]
    cls_path = os.path.join(train_dir, cls)
    c1_files = sorted(glob.glob(os.path.join(cls_path, "*_c1.npy")))[:3]
    for c1f in c1_files:
        c2f = c1f.replace("_c1.npy", "_c2.npy")
        if os.path.exists(c2f):
            c1 = np.load(c1f)
            c2 = np.load(c2f)
            diff = np.mean(np.abs(c1 - c2))
            print(f"  {os.path.basename(c1f)} vs c2: mean diff = {diff:.4f} "
                  f"({'IDENTICAL - bad!' if diff < 0.001 else 'different - good'})")
        else:
            print(f"  No c2 file found for {os.path.basename(c1f)}")

# ============================================================
# Check 5: Quick overfit test — can model memorise 1 batch?
# ============================================================
print("\n" + "=" * 60)
print("CHECK 5: Sanity — can the model overfit a single batch?")
print("=" * 60)
print("  Running mini overfit test (10 samples, 50 epochs)...")

import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.preprocessing import LabelEncoder

# Load just 10 samples
X_small, y_small_str = [], []
if os.path.exists(train_dir):
    for cls in sorted(os.listdir(train_dir), key=lambda x: float(x)):
        cls_path = os.path.join(train_dir, cls)
        if not os.path.isdir(cls_path):
            continue
        files = glob.glob(os.path.join(cls_path, "*.npy"))[:1]
        for f in files:
            arr = np.load(f)
            # Clip extreme values and re-normalise to [0,1]
            arr = np.clip(arr, -5, 5)
            arr = (arr - arr.min()) / (arr.max() - arr.min() + 1e-8)
            X_small.append(arr)
            y_small_str.append(cls)

X_small = np.array(X_small, dtype=np.float32)
enc = LabelEncoder()
enc.fit(sorted(set(y_small_str), key=lambda x: float(x)))
y_small = enc.transform(y_small_str)

print(f"  Mini dataset: {X_small.shape}, classes: {enc.classes_}")
print(f"  Value range: [{X_small.min():.3f}, {X_small.max():.3f}]")

# Tiny model
inp = layers.Input(shape=X_small.shape[1:])
x   = layers.Flatten()(inp)
x   = layers.Dense(64, activation='relu')(x)
out = layers.Dense(len(enc.classes_), activation='softmax')(x)
mini_model = models.Model(inp, out)
mini_model.compile(optimizer='adam',
                   loss='sparse_categorical_crossentropy',
                   metrics=['accuracy'])

hist = mini_model.fit(X_small, y_small, epochs=50, verbose=0)
final_acc = hist.history['accuracy'][-1]
print(f"  Final accuracy on 10 samples after 50 epochs: {final_acc*100:.1f}%")
if final_acc > 0.9:
    print("  RESULT: Model CAN learn → problem is in preprocessing/data quality")
else:
    print("  RESULT: Model CANNOT learn even tiny data → check NaN/Inf values above")