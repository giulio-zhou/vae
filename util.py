from sklearn.datasets import fetch_mldata
from PIL import Image
import numpy as np

def get_mnist_data():
    mnist_path = 'mnist'
    mnist = fetch_mldata('MNIST original', data_home=mnist_path)
    return mnist['data'].reshape(-1, 28, 28), mnist['target']

def get_orl_face_data():
    root_dir = 'orl_faces'
    imgs, labels = [], []
    for subject_id in range(40):
        for img_idx in range(10):
            img_path = '%s/s%d/%d.pgm' % (root_dir, subject_id + 1, img_idx + 1)
            print(img_path)
            img = np.array(Image.open(img_path))
            imgs.append(img)
            labels.append(subject_id)
    return np.array(imgs), np.array(labels)
