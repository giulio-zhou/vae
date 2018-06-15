from sklearn.datasets import fetch_mldata, fetch_olivetti_faces
from PIL import Image
import h5py
import numpy as np

def get_mnist_data():
    mnist_path = 'mnist'
    mnist = fetch_mldata('MNIST original', data_home=mnist_path)
    return mnist['data'].reshape(-1, 28, 28), mnist['target']

def get_olivetti_data():
    olivetti_path = 'olivetti'
    face_data = fetch_olivetti_faces(olivetti_path)
    return face_data.images, face_data.target

def get_train_data():
    path = '/home/giulio/train_cam_full_64x64.h5'
    hdf5_file = h5py.File(path, 'r')
    data, labels = hdf5_file['data'], None
    return data, labels

def get_jh_data():
    path = '/home/giulio/new_jh_1_and_3_64x64_5fps.h5'
    hdf5_file = h5py.File(path, 'r')
    data, labels = hdf5_file['data'], hdf5_file['labels']
    return data, labels
