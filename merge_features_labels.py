import pandas as pd

# Read files
df_features = pd.read_csv("features_train_full.csv")
df_labels = pd.read_csv("hemorrhage_labels_full.csv")

# Merge on filename
df_final = pd.merge(df_features, df_labels, on="filename", how="inner")

# Save the final file
df_final.to_csv("features_labels_train_full.csv", index=False)
print(f"✅ Final merged file created: features_labels_train_full.csv")
print(f"Number of rows in the final file: {len(df_final)}")
