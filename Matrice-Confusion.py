import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt

# 1️⃣ قراءة CSV
df = pd.read_csv("features_labels_train_full.csv")  # غيري الاسم حسب ملفك

# 2️⃣ اختيار الميزات و label
features = ['lbp_mean','lbp_var','contrast','energy','homogeneity','correlation']
X = df[features].values
y = df['label'].map({'Normal':0, 'Hemorrhage':1}).values  # تحويل النصوص لأرقام

# 3️⃣ تقسيم البيانات
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# 4️⃣ تدريب نموذج Random Forest
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# 5️⃣ التنبؤ على مجموعة الاختبار
y_pred = model.predict(X_test)

# 6️⃣ مصفوفة الارتباك
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6,5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Normal','Hemorrhage'], 
            yticklabels=['Normal','Hemorrhage'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()

# 7️⃣ Precision, Recall, F1
print(classification_report(y_test, y_pred, target_names=['Normal','Hemorrhage']))
