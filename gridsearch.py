import pandas as pd
import numpy as np
import cv2 as cv
import os
import time

from cuml.metrics import accuracy_score
from cuml.model_selection import train_test_split
from cuml.preprocessing import StandardScaler
from cuml.pipeline import Pipeline
from cuml.svm import SVC
from skimage.feature import hog
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import GridSearchCV

class Preprocessor(BaseEstimator, TransformerMixin):
    def __init__(self):
        super().__init__()
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        features = []
        for image in X:
            r, g, b = cv.split(image)
            hog_r = hog(r, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(2, 2), visualize=False)
            hog_g = hog(g, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(2, 2), visualize=False)
            hog_b = hog(b, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(2, 2), visualize=False)
            hog_features = np.concatenate((hog_r, hog_g, hog_b))
            
            image = cv.resize(image, (16,16))
            feature = np.hstack((image.flatten(), hog_features))
            features.append(feature)
        features = np.array(features)
        print(features.shape)
        return features
    
print()
dir = os.getcwd()
df = pd.read_csv('train.csv')

images = []
labels = []

for index, row in df.iterrows():
    img_path = os.path.join(dir, 'train_ims', row['im_name'])
    try:
        image = cv.imread(img_path, cv.IMREAD_ANYCOLOR)
    except:
        print(f"image {img_path} not found")
    images.append(cv.flip(image, 1))
    images.append(cv.flip(image, 0))
    images.append(cv.flip(image, -1))
    images.append(image)
    for i in range(4): labels.append(row['label'])

images = np.array(images)
labels = np.array(labels)

preprocessor = Preprocessor()
images = preprocessor.fit_transform(images)
scl = StandardScaler()
images = scl.fit_transform(images)

X_train, X_test, y_train, y_test = train_test_split(images, labels, test_size=0.2, random_state=42)

svc = SVC(gamma='scale')
param_grid = {
    'C': [0.1, 1, 10, 100],
    'kernel': ['rbf', 'poly', 'sigmoid']
}
gc = GridSearchCV(estimator=svc, param_grid=param_grid, verbose=3, scoring='accuracy', refit=True, cv=3)

start_time = time.time()
gc.fit(X_train, y_train)
end_time = time.time()

print("Full training completed. (GridSearchCV)")
# print(f"Training time: {end_time - start_time} seconds")
print()

print(f"Best parameters: {gc.best_params_}")
print(f"Best score: {gc.best_score_}")
print()

model = gc.best_estimator_

y_test_pred = model.predict(X_test)
print(f"Test score: {accuracy_score(y_test, y_test_pred)}")
print()

test_df = pd.read_csv('test.csv')
test_images = []
test_names = []

for index, row in test_df.iterrows():
    img_path = os.path.join(dir, 'test_ims', row['im_name'])
    try:
        image = cv.imread(img_path, cv.IMREAD_ANYCOLOR)
    except:
        print(f"image {img_path} not found")
    test_images.append(image)
    test_names.append(row['im_name'])

X_test = np.array(test_images)

start_time = time.time()
y_test_pred = model.predict(X_test)
end_time = time.time()
print("テストデータの予測は完了しました。")
print(f"Prediction time: {end_time - start_time} seconds")
print()

output_df = pd.DataFrame({
    'im_name': test_names,
    'label': y_test_pred
})
output_df.to_csv('grid_submission.csv', index=False)
print("grid_submission.csv saved.")
print()

# Save the model
# import joblib
# joblib.dump(pipeline, 'model.pkl')