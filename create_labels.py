import pandas as pd

# مثال صغير فقط
data = {
    "filename": ["049_slice0.png", "049_slice1.png", "050_slice0.png"],
    "label": ["IPH", "IPH", "Normal"]
}

labels = pd.DataFrame(data)
labels.to_csv("hemorrhage_labels.csv", index=False)
print("✅ تم إنشاء ملف labels")
