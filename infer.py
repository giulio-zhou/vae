from run import VAE, ConditionalVAE
from run import vae
from util import get_mnist_data
import numpy as np
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

if __name__ == '__main__':
    compare_mean_vs_samples()
