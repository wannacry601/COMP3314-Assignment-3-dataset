import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import cv2
import os

# load the data
data_dir = os.getcwd()
train_csv_path = os.path.join(data_dir, 'train.csv')
train_images_dir = os.path.join(data_dir, 'train_ims')

train_df = pd.read_csv(os.path.join(data_dir, 'train.csv'))
test_df = pd.read_csv(os.path.join(data_dir, 'test.csv'))

from sklearn.model_selection import train_test_split

IMG_SIZE = (64, 64)

# Initialize lists to hold image data and labels
X = []  # Feature vectors (flattened grayscale images)
y = []  # Labels

for idx, row in train_df.iterrows():
    im_name = row['im_name']
    label = row['label']
    image_path = os.path.join(train_images_dir, im_name)
    
    try:
        img_gray = cv2.imread(image_path,cv2.IMREAD_GRAYSCALE)
        img_gray = cv2.resize(img_gray, IMG_SIZE)
        X.append(img_gray.flatten())
        y.append(label)
    except:
        raise Exception(f"Image {im_name} does not exist.")

X = np.array(X)
y = np.array(y)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4)
print(y_train.shape)

from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.ensemble import AdaBoostClassifier
import time

svm = SVC(C=1,kernel="rbf")
ada = AdaBoostClassifier(estimator=svm, n_estimators=10, learning_rate=0.1)

time1 = time.time()
ada = ada.fit(X_train, y_train)
time2 = time.time()
y_train_pred = ada.predict(X_train)
y_test_pred = ada.predict(y_test)

ada_test = accuracy_score(y_test, y_test_pred)
print(ada_test)
print(time2-time1)