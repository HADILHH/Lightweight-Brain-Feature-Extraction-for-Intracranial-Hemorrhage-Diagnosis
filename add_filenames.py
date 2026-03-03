# ملف: add_filenames_auto.py
import pandas as pd
import os

# تحميل الميزات
features = pd.read_csv("features_train.csv")

# المجلد اللي فيه صور train
train_dir = "data/train"  # غيّري حسب مسارك الفعلي

# ترتيب الصور حسب الاسم
image_files = sorted(os.listdir(train_dir))

# تأكد أن عدد الصور يساوي عدد الصفوف في features
if len(image_files) != len(features):
    raise ValueError(f"عدد الصور ({len(image_files)}) لا يساوي عدد الصفوف ({len(features)})")

# إضافة العمود
features['filename'] = image_files

# حفظ الملف الجديد
features.to_csv("features_train_with_filenames.csv", index=False)
print("✅ تم إنشاء ملف features_train_with_filenames.csv مع العمود filename")
