from run import VAE, ConditionalVAE
from run import vae
from skimage import img_as_ubyte
from skimage.transform import resize
import numpy as np
import os
import skimage.io as skio

def compare_mean_vs_samples():
    data, labels = get_mnist_data() 
    data = data.reshape(-1, 784) / 255.
    # Settings.
    model_dir = 'out/model@10000'
    img_height, img_width = 28, 28
    input_dim = 784
    latent_dim = 10
    num_classes = 10
    num_samples = 10
    output_path = 'mean_vs_samples.png'
    # Model.
    vae_model = VAE(vae, input_dim, latent_dim)
    vae_model.load_model(model_dir)
    # Create visualization.
    output_buffer = np.zeros((img_height * num_classes,
                              img_width * (num_samples + 2)))
    for c in range(num_classes):
        idx = np.random.choice(np.where(labels == c)[0])
        X = np.expand_dims(data[idx], axis=0)
        X_square = X.reshape(img_height, img_width)
        encoded_X_mean = vae_model.encode(X)
        decoded_X_mean = vae_model.decode(encoded_X_mean)[0]
        decoded_X_mean = decoded_X_mean.reshape(img_height, img_width)
        # Draw.
        output_buffer[c*img_height:(c+1)*img_height, :img_width] = X_square
        output_buffer[c*img_height:(c+1)*img_height,
                      img_width:2*img_width] = decoded_X_mean
        for j in range(num_samples):
            decoded_X_sample = vae_model.encode_decode_sample(X)[0]
            decoded_X_sample = decoded_X_sample.reshape(img_height, img_width)
            output_buffer[c*img_height:(c+1)*img_height,
                          (j+2)*img_width:(j+3)*img_width] = decoded_X_sample
    skio.imsave(output_path, output_buffer)

def interpolate_between_classes():
    data, labels = get_mnist_data() 
    data = data.reshape(-1, 784) / 255.
    # Settings.
    model_dir = 'out/model@10000'
    img_height, img_width = 28, 28
    input_dim = 784
    latent_dim = 10
    num_classes = 10
    num_steps = 10
    output_path = 'interpolate_classes.png'
    # Model.
    vae_model = VAE(vae, input_dim, latent_dim)
    vae_model.load_model(model_dir)
    # Data to examine.
    class_order = np.arange(num_classes)
    np.random.shuffle(class_order)
    class_idx = [np.random.choice(np.where(labels == c)[0]) \
                 for c in range(num_classes)]
    xs = vae_model.encode(data[class_idx])
    # Create visualization.
    output_buffer = np.zeros((img_height * (num_classes - 1),
                              img_width * num_steps))
    for c in range(num_classes-1):
        step = (xs[c+1] - xs[c]) / float(num_steps)
        X = np.expand_dims(xs[c], axis=0)
        for i in range(num_steps):
            decoded_X = vae_model.decode(X + i*step)[0]
            decoded_X = decoded_X.reshape(img_height, img_width)
            output_buffer[c*img_height:(c+1)*img_height,
                          i*img_width:(i+1)*img_width] = decoded_X
    skio.imsave(output_path, output_buffer)

def generate_encodings():
    data, labels = get_mnist_data() 
    data = data.reshape(-1, 784) / 255.
    # Settings.
    model_dir = 'out/model@10000'
    img_height, img_width = 28, 28
    input_dim = 784
    latent_dim = 10
    num_classes = 10
    batch_size = 100
    output_dir = 'vae_encodings'
    # Model.
    vae_model = VAE(vae, input_dim, latent_dim)
    vae_model.load_model(model_dir)
    # Perform encoding.
    encoded_data = []
    for i in range(0, len(data), batch_size):
        start, end = i, min(i + batch_size, len(data))
        X_batch = data[start:end]
        encoded_inputs = vae_model.encode(X_batch)
        encoded_data.append(encoded_inputs)    
    encoded_data = np.concatenate(encoded_data, axis=0)
    # Save data.
    rgb_data = np.stack([data.reshape(-1, 28, 28)] * 3, axis=-1)
    rgb_data = np.array([img_as_ubyte(resize(x, (14, 14))) for x in rgb_data])
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    np.save(output_dir + '/features.npy', encoded_data)
    np.save(output_dir + '/data.npy', rgb_data)
    np.save(output_dir + '/labels.npy', labels)

def classification_accuracy():
    from sklearn.svm import LinearSVC
    from sklearn.neural_network import MLPClassifier
    from sklearn.metrics import accuracy_score
    data, labels = get_mnist_data() 
    data = data.reshape(-1, 784) / 255.
    # Settings.
    model_dir = 'out/model@10000'
    img_height, img_width = 28, 28
    input_dim = 784
    latent_dim = 10
    num_classes = 10
    batch_size = 100
    output_dir = 'vae_encodings'
    # Model.
    vae_model = VAE(vae, input_dim, latent_dim)
    vae_model.load_model(model_dir)
    # Perform encoding.
    encoded_data = []
    for i in range(0, len(data), batch_size):
        start, end = i, min(i + batch_size, len(data))
        X_batch = data[start:end]
        encoded_inputs = vae_model.encode(X_batch)
        encoded_data.append(encoded_inputs)    
    encoded_data = np.concatenate(encoded_data, axis=0)
    # Test.
    idx = np.arange(len(data))
    np.random.shuffle(idx)
    train_idx, test_idx = idx[:60000], idx[60000:]
    # model = LinearSVC(verbose=2)
    model = MLPClassifier(hidden_layer_sizes=[10, 10, 10, 10], verbose=2)
    X_train, y_train = encoded_data[train_idx], labels[train_idx]
    X_test, y_test = encoded_data[test_idx], labels[test_idx]
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    print(accuracy_score(preds, y_test))

if __name__ == '__main__':
    compare_mean_vs_samples()
    interpolate_between_classes()
    generate_encodings()
    classification_accuracy()
