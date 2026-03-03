import os
import numpy as np
import pandas as pd
from skimage.io import imread
from skimage.feature import local_binary_pattern

# 1️⃣ المسارات
ct_path = "data/ct_slices"   # مسار كل الـ slices
mask_path = "data/masks"    # مسار كل الـ masks
output_features = "features_train_full.csv"
output_labels = "hemorrhage_labels_full.csv"
output_final = "features_labels_train_full.csv"

# 2️⃣ إعداد الـ LBP
radius = 1
n_points = 8 * radius

# 3️⃣ استخراج الميزات لكل slice
features_list = []
labels_list = []

ct_files = sorted(os.listdir(ct_path))

for ct_file in ct_files:
    ct_img = imread(os.path.join(ct_path, ct_file))
    mask_file_path = os.path.join(mask_path, ct_file)
    
    if os.path.exists(mask_file_path):
        mask_img = imread(mask_file_path)
        ct_masked = ct_img * (mask_img > 0)
        label = "Hemorrhage"  # حسب الماسك
    else:
        print(f"⚠️ لم يتم العثور على ملف الماسك: {mask_file_path}, سيتم استخدام الصورة كاملة واعتبارها Normal")
        ct_masked = ct_img
        label = "Normal"
    
    # حساب LBP
    if ct_masked.ndim != 2:
        ct_masked = ct_masked.squeeze()  # للتأكد أنها 2D
    lbp = local_binary_pattern(ct_masked, n_points, radius, method="uniform")
    lbp_mean = lbp.mean()
    lbp_var = lbp.var()
    
    features_list.append({
        "filename": ct_file,
        "lbp_mean": lbp_mean,
        "lbp_var": lbp_var
    })
    labels_list.append({
        "filename": ct_file,
        "label": label
    })

# حفظ الميزات
df_features = pd.DataFrame(features_list)
df_features.to_csv(output_features, index=False)
print(f"✅ تم استخراج الميزات وحفظها في {output_features}")

# حفظ التسميات
df_labels = pd.DataFrame(labels_list)
df_labels.to_csv(output_labels, index=False)
print(f"✅ تم إنشاء ملف التسميات: {output_labels}")

# دمج الميزات مع التسميات
df_final = pd.merge(df_features, df_labels, on="filename")
df_final.to_csv(output_final, index=False)
print(f"✅ تم إنشاء الملف النهائي بعد الدمج: {output_final}")
