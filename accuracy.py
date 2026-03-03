import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# تحميل الموديل
model = joblib.load("ich_model.pkl")

# تحميل البيانات
df = pd.read_csv("features_labels_train_full.csv")

# حذف عمود اسم الصورة
df = df.drop(columns=['filename'])

# تحويل الـ labels من نص إلى أرقام
df['label'] = df['label'].map({
    'Normal': 0,
    'Hemorrhage': 1
})

# تأكدي ما بقاش قيم NaN
df = df.dropna(subset=['label'])

# فصل الخصائص والـ label
X = df.drop('label', axis=1)
y = df['label']

# تقسيم البيانات
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# التنبؤ
y_pred = model.predict(X_test.values)

# حساب الدقة
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy of ICH model: {accuracy:.4f}")
