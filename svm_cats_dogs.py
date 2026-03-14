import os
import cv2
import numpy as np
import random

from skimage.feature import hog
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score


# Dataset folder
data_path = "train"

# Image size
img_size = 64

X = []
y = []

# Load image names
images = os.listdir(data_path)
random.shuffle(images)

# Limit dataset size for faster training
images = images[:2000]


for img in images:

    img_path = os.path.join(data_path, img)

    # Read grayscale image
    image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

    if image is None:
        continue

    # Resize image
    image = cv2.resize(image, (img_size, img_size))

    # Extract HOG features
    features = hog(
        image,
        orientations=9,
        pixels_per_cell=(8,8),
        cells_per_block=(2,2),
        visualize=False
    )

    X.append(features)

    # Label
    if "cat" in img.lower():
        y.append(0)
    else:
        y.append(1)


# Convert to numpy
X = np.array(X)
y = np.array(y)

print("Dataset shape:", X.shape)
print("Classes:", set(y))


# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


# Train SVM
model = SVC(kernel="linear")

print("Training model...")
model.fit(X_train, y_train)


# Predictions
predictions = model.predict(X_test)

accuracy = accuracy_score(y_test, predictions)

print("Model Accuracy:", accuracy)