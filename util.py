from sklearn.datasets import fetch_mldata, fetch_olivetti_faces
from PIL import Image
import numpy as np

def get_mnist_data():
    mnist_path = 'mnist'
    mnist = fetch_mldata('MNIST original', data_home=mnist_path)
    return mnist['data'].reshape(-1, 28, 28), mnist['target']

def get_olivetti_data():
    olivetti_path = 'olivetti'
    face_data = fetch_olivetti_faces(olivetti_path)
    return face_data.images, face_data.target
