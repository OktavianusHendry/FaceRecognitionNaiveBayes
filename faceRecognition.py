import numpy as np
from sklearn.naive_bayes import GaussianNB
from sklearn.decomposition import PCA
import cv2
import os

# Fungsi untuk membaca dataset wajah
def load_dataset(path):
    X = []
    y = []
    labels = os.listdir(path)
    for label in labels:
        label_path = os.path.join(path, label)
        if not os.path.isdir(label_path):
            continue
        images = os.listdir(label_path)
        for image_name in images:
            image_path = os.path.join(label_path, image_name)
            image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            if image is not None:
                X.append(image)
                y.append(label)
    return np.array(X), np.array(y)

# Fungsi untuk mengekstraksi fitur wajah menggunakan PCA
def extract_features(X):
    num_samples = X.shape[0]
    X = X.reshape(num_samples, -1)
    pca = PCA(n_components=0.95)
    X = pca.fit_transform(X)
    return X

# Fungsi untuk melatih model Naive Bayes
def train_naive_bayes(X_train, y_train):
    model = GaussianNB()
    model.fit(X_train, y_train)
    return model

# Fungsi untuk melakukan pengujian pada model
def test_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    accuracy = np.mean(y_pred == y_test)
    return accuracy

# Path dataset wajah
dataset_path = 'dataset_faces'

# Memuat dataset wajah
X, y = load_dataset(dataset_path)

# Mengubah dataset menjadi fitur-fitur wajah
X_features = extract_features(X)

# Memisahkan dataset menjadi set pelatihan dan pengujian
split_ratio = 0.8
split_index = int(split_ratio * X_features.shape[0])
X_train, X_test = X_features[:split_index], X_features[split_index:]
y_train, y_test = y[:split_index], y[split_index:]

# Melatih model Naive Bayes
model = train_naive_bayes(X_train, y_train)

# Menguji model
accuracy = test_model(model, X_test, y_test)
print('Akurasi: {:.2f}%'.format(accuracy * 100))
