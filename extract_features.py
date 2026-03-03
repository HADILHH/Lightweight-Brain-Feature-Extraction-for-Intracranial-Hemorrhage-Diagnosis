import os
import cv2
import numpy as np
import pandas as pd
import pywt
from skimage.feature import local_binary_pattern, graycomatrix, graycoprops

data_path = "data/train"
output_csv = "features_train.csv"

features_list = []
labels_list = []

for img_name in os.listdir(data_path):
    if not img_name.lower().endswith(".png"):
        continue  # نتجاهل أي ملف غير صورة
    img_path = os.path.join(data_path, img_name)
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        print(f"❌ فشل قراءة الصورة {img_name}")
        continue

    # --- هنا مثال لاستخراج بعض الميزات ---
    lbp = local_binary_pattern(img, P=8, R=1, method='uniform')
    lbp_mean = lbp.mean()
    lbp_var = lbp.var()

    features = [lbp_mean, lbp_var]

    # Label مؤقت
    label = "unknown"

    features_list.append(features)
    labels_list.append(label)
    print(f"✔ تمت معالجة {img_name}")

# حفظ CSV
df = pd.DataFrame(features_list, columns=["lbp_mean", "lbp_var"])
df["label"] = labels_list
df.to_csv(output_csv, index=False)
print(f"✔ تم حفظ الميزات في {output_csv}")
