import pandas as pd

data = pd.read_csv("features_labels_train.csv")
print("أسماء الأعمدة في الملف:")
print(data.columns)
