from sklearn.datasets import fetch_mldata

def get_mnist_data():
  mnist_path = 'mnist'
  mnist = fetch_mldata('MNIST original', data_home=mnist_path)
  return mnist['data'].reshape(-1, 28, 28), mnist['target']
