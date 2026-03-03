import os
import nibabel as nib
from PIL import Image
import numpy as np

dataset_path = "data/ct_scans"
output_path = "data/ct_slices"

os.makedirs(output_path, exist_ok=True)

files = os.listdir(dataset_path)
print(f"عدد الملفات الكلي: {len(files)}")

# Batch processing لكل الملفات
for idx, f in enumerate(files):
    try:
        img_path = os.path.join(dataset_path, f)
        nii_img = nib.load(img_path)
        data = nii_img.get_fdata()
        num_slices = data.shape[2]

        print(f"\nمعالجة الملف {idx+1}/{len(files)}: {f}, عدد الشرائح: {num_slices}")

        for i in range(num_slices):
            slice_2d = data[:, :, i]
            slice_norm = (slice_2d - np.min(slice_2d)) / (np.max(slice_2d) - np.min(slice_2d)) * 255
            slice_norm = slice_norm.astype(np.uint8)

            slice_img = Image.fromarray(slice_norm)
            slice_name = f"{os.path.splitext(f)[0]}_slice_{i:03d}.png"
            slice_img.save(os.path.join(output_path, slice_name))

        print(f"✔ تم حفظ جميع الشرائح للملف {f}")

    except Exception as e:
        print(f"❌ فشل معالجة الملف {f}: {e}")

print("\n🎉 جميع الملفات تم تحويلها إلى PNG بنجاح!")
