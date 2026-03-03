import cv2
import joblib
import numpy as np
from skimage.feature import local_binary_pattern


model = joblib.load("model/ich_model.pkl")
scaler = joblib.load("model/scaler.pkl")


img = cv2.imread("test.png", 0)


lbp = local_binary_pattern(img, 8, 1, method='uniform')
features = np.array([[lbp.mean(), lbp.var()]])
features = scaler.transform(features)


pred = model.predict(features)[0]


print("Prediction:", "Hemorrhage" if pred == 1 else "Normal")