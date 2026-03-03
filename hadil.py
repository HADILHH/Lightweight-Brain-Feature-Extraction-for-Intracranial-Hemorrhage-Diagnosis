import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
from sklearn.model_selection import learning_curve, train_test_split

# -----------------------------
# 1️⃣ تحميل البيانات
# -----------------------------
df = pd.read_csv("features_labels_train_full.csv")  # غيري الاسم حسب ملفك

features = ['lbp_mean','lbp_var','contrast','energy','homogeneity']  # أو جميع الميزات اللي عندك
X = df[features]
y = df['label']  # 0 = Normal, 1 = Hemorrhage

# تقسيم البيانات إذا لم يكن جاهز
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=423/2392, random_state=42, stratify=y
)

# -----------------------------
# 2️⃣ تحميل scaler وموديل RF المدرب
# -----------------------------
with open("scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

with open("ich_model.pkl", "rb") as f:
    rf_model = pickle.load(f)

X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)

# -----------------------------
# 3️⃣ تعريف باقي الموديلات baseline
# -----------------------------
models = {
    "Random Forest": rf_model,
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
    "KNN": KNeighborsClassifier(),
    "SVM": SVC(probability=True, random_state=42)
}

# تدريب الباسلاين إذا لم يكن مدرب مسبقاً
for name, model in models.items():
    if name != "Random Forest":  # RF مدرب مسبقاً
        model.fit(X_train_scaled, y_train)

# -----------------------------
# 4️⃣ تقييم كل الموديلات وحساب Metrics
# -----------------------------
results = []

for name, model in models.items():
    y_pred = model.predict(X_test_scaled)
    y_proba = model.predict_proba(X_test_scaled)[:,1] if hasattr(model, "predict_proba") else None
    
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_proba) if y_proba is not None else np.nan
    
    results.append([name, acc, prec, rec, f1, auc])

# -----------------------------
# 5️⃣ إنشاء Table X
# -----------------------------
table_X = pd.DataFrame(results, columns=["Model","Accuracy","Precision","Recall","F1-score","AUC"])
print("\nTable X: Performance Comparison of Models\n")
print(table_X)

# -----------------------------
# 6️⃣ رسم Learning Curve للـ Random Forest
# -----------------------------
train_sizes, train_scores, val_scores = learning_curve(
    rf_model, X_train_scaled, y_train, cv=5,
    train_sizes=np.linspace(0.1, 1.0, 5), scoring='accuracy'
)

train_mean = np.mean(train_scores, axis=1)
val_mean = np.mean(val_scores, axis=1)

plt.figure(figsize=(8,6))
plt.plot(train_sizes, train_mean, label='Training Accuracy', marker='o')
plt.plot(train_sizes, val_mean, label='Validation Accuracy', marker='o')
plt.title('Learning Curve - Random Forest')
plt.xlabel('Training Set Size')
plt.ylabel('Accuracy')
plt.grid(True)
plt.legend()
plt.show()

# -----------------------------
# 7️⃣ Confusion Matrix للـ Random Forest
# -----------------------------
y_pred_rf = rf_model.predict(X_test_scaled)
cm = confusion_matrix(y_test, y_pred_rf)
plt.figure(figsize=(5,4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Normal','Hemorrhage'], yticklabels=['Normal','Hemorrhage'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix - Random Forest')
plt.show()
