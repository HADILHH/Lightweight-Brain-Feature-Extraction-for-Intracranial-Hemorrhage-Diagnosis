import os
import cv2
import numpy as np
import joblib
from skimage.feature import local_binary_pattern, graycomatrix, graycoprops

# =============================
# Set the image path
# =============================
image_path = "data/test/test.png"

if not os.path.exists(image_path):
    raise FileNotFoundError(f"❌ Image not found: {image_path}")

# =============================
# Load image and convert to grayscale
# =============================
img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
if img is None:
    raise ValueError(f"❌ OpenCV cannot read the image: {image_path}")

print(f"✅ Image loaded successfully, shape: {img.shape}")

# =============================
# Extract LBP features
# =============================
radius = 1
n_points = 8 * radius
lbp = local_binary_pattern(img, n_points, radius, method="uniform")
lbp_mean = lbp.mean()
lbp_var = lbp.var()

# =============================
# Extract GLCM features
# =============================
glcm = graycomatrix(img, distances=[1], angles=[0], symmetric=True, normed=True)
contrast = graycoprops(glcm, 'contrast')[0,0]
energy = graycoprops(glcm, 'energy')[0,0]
homogeneity = graycoprops(glcm, 'homogeneity')[0,0]
correlation = graycoprops(glcm, 'correlation')[0,0]

# =============================
# Combine features into an array
# =============================
X_new = np.array([[lbp_mean, lbp_var, contrast, energy, homogeneity, correlation]])

# =============================
# Load the model and scaler
# =============================
model_path = "model/ich_model.pkl"
scaler_path = "model/scaler.pkl"

if not os.path.exists(model_path) or not os.path.exists(scaler_path):
    raise FileNotFoundError("❌ Make sure the model and scaler exist in the 'model' folder")

model = joblib.load(model_path)
scaler = joblib.load(scaler_path)

# =============================
# Apply StandardScaler
# =============================
X_new_scaled = scaler.transform(X_new)

# =============================
# Predict
# =============================
pred = model.predict(X_new_scaled)[0]
pred_label = "Normal" if pred == 0 else "Hemorrhage"

print(f"\n📊 Result: {pred_label}")
