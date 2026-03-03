import os
import cv2
import numpy as np
import joblib
import pandas as pd
import tkinter as tk
from tkinter import filedialog, messagebox
from skimage.feature import local_binary_pattern, graycomatrix, graycoprops

# Load model and scaler
model_path = "model/ich_model.pkl"
scaler_path = "model/scaler.pkl"
model = joblib.load(model_path)
scaler = joblib.load(scaler_path)

# Setup Tkinter
root = tk.Tk()
root.title("ICH Detection")
root.geometry("400x200")

def predict_image():
    file_path = filedialog.askopenfilename(filetypes=[("PNG Images","*.png"),("All files","*.*")])
    if not file_path:
        return

    img = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        messagebox.showerror("Error", "❌ Cannot read the image with OpenCV")
        return

    # Extract LBP features
    radius = 1
    n_points = 8 * radius
    lbp = local_binary_pattern(img, n_points, radius, method="uniform")
    lbp_mean = lbp.mean()
    lbp_var = lbp.var()

    # Extract GLCM features
    glcm = graycomatrix(img, distances=[1], angles=[0], symmetric=True, normed=True)
    contrast = graycoprops(glcm, 'contrast')[0,0]
    energy = graycoprops(glcm, 'energy')[0,0]
    homogeneity = graycoprops(glcm, 'homogeneity')[0,0]
    correlation = graycoprops(glcm, 'correlation')[0,0]

    # Create DataFrame for features
    feature_names = ["lbp_mean", "lbp_var", "contrast", "energy", "homogeneity", "correlation"]
    X_new = pd.DataFrame([[lbp_mean, lbp_var, contrast, energy, homogeneity, correlation]], columns=feature_names)

    # Apply StandardScaler
    X_new_scaled = scaler.transform(X_new)

    # Prediction
    pred = model.predict(X_new_scaled)[0]
    pred_label = "Normal" if pred == 0 else "Hemorrhage"

    messagebox.showinfo("Prediction", f"📊 Result: {pred_label}")

# Button to select image
btn = tk.Button(root, text="Select Image for Prediction", command=predict_image)
btn.pack(pady=50)

root.mainloop()
