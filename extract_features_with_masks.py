import os
import numpy as np
import pandas as pd
from skimage.io import imread
from skimage.feature import local_binary_pattern, graycomatrix, graycoprops

# =============================
# المسارات
# =============================
CT_PATH = "data/ct_slices"
MASK_PATH = "data/masks"
NII_LABELS_FILE = "hemorrhage_labels.csv"

OUTPUT_FEATURES = "features_train_full.csv"
OUTPUT_LABELS = "hemorrhage_labels_full.csv"
OUTPUT_FINAL = "features_labels_train_full.csv"

# =============================
# LBP parameters
# =============================
radius = 1
n_points = 8 * radius

# =============================
# قراءة labels على مستوى .nii
# =============================
nii_labels = pd.read_csv(NII_LABELS_FILE)

def get_label_from_slice(filename):
    """
    مثال: 078_slice_012.png → 078.nii
    """
    nii_name = filename.split("_")[0] + ".nii"
    row = nii_labels[nii_labels["filename"] == nii_name]
    if len(row) == 0:
        return "Normal"
    return row.iloc[0]["label"]

# =============================
# استخراج الميزات
# =============================
features = []
labels = []

ct_files = sorted(os.listdir(CT_PATH))

for ct_file in ct_files:
    ct_path = os.path.join(CT_PATH, ct_file)
    mask_path = os.path.join(MASK_PATH, ct_file)

    ct_img = imread(ct_path)

    # تأكد grayscale
    if ct_img.ndim == 3:
        ct_img = ct_img[:, :, 0]

    # إذا الماسك موجود
    if os.path.exists(mask_path):
        mask_img = imread(mask_path)
        if mask_img.ndim == 3:
            mask_img = mask_img[:, :, 0]
        ct_used = ct_img * (mask_img > 0)
    else:
        print(f"⚠️ لم يتم العثور على ملف الماسك: {mask_path}, سيتم استخدام الصورة كاملة")
        ct_used = ct_img

    # =============================
    # LBP
    # =============================
    lbp = local_binary_pattern(ct_used, n_points, radius, method="uniform")
    lbp_mean = lbp.mean()
    lbp_var = lbp.var()

    # =============================
    # GLCM
    # =============================
    ct_uint8 = np.clip(ct_used, 0, 255).astype(np.uint8)

    glcm = graycomatrix(
        ct_uint8,
        distances=[1],
        angles=[0],
        levels=256,
        symmetric=True,
        normed=True
    )

    contrast = graycoprops(glcm, "contrast")[0, 0]
    energy = graycoprops(glcm, "energy")[0, 0]
    homogeneity = graycoprops(glcm, "homogeneity")[0, 0]
    correlation = graycoprops(glcm, "correlation")[0, 0]

    # =============================
    # حفظ
    # =============================
    features.append({
        "filename": ct_file,
        "lbp_mean": lbp_mean,
        "lbp_var": lbp_var,
        "contrast": contrast,
        "energy": energy,
        "homogeneity": homogeneity,
        "correlation": correlation
    })

    labels.append({
        "filename": ct_file,
        "label": get_label_from_slice(ct_file)
    })

# =============================
# حفظ الملفات
# =============================
df_features = pd.DataFrame(features)
df_labels = pd.DataFrame(labels)

df_features.to_csv(OUTPUT_FEATURES, index=False)
df_labels.to_csv(OUTPUT_LABELS, index=False)

df_final = pd.merge(df_features, df_labels, on="filename")
df_final.to_csv(OUTPUT_FINAL, index=False)

print("✅ تم استخراج الميزات")
print("✅ تم إنشاء labels")
print("✅ تم إنشاء الملف النهائي:", OUTPUT_FINAL)
