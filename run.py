from skimage.transform import resize
from util import get_mnist_data, get_olivetti_data
import numpy as np
import os
import skimage.io as skio
import tensorflow as tf

def mnist_decoders(data_dim):
    dec_fc1 = tf.layers.Dense(units=200, activation=tf.nn.relu)
    dec_fc2 = tf.layers.Dense(units=500, activation=tf.nn.relu)
    dec_fc3 = tf.layers.Dense(units=data_dim, activation=tf.nn.sigmoid)
    return [dec_fc1, dec_fc2, dec_fc3]

def mnist_conv_decoders():
    dec_fc1 = tf.layers.Dense(units=200, activation=tf.nn.relu)
    dec_fc2 = tf.layers.Dense(units=7*7*32, activation=tf.nn.relu)
    dec_conv1 = tf.layers.Conv2DTranspose(filters=16, kernel_size=(3, 3), strides=2,
                                          padding='same', activation=tf.nn.relu)
    dec_conv2 = tf.layers.Conv2DTranspose(filters=1, kernel_size=(3, 3), strides=2,
                                          padding='same', activation=tf.nn.sigmoid)
    dec_fc, dec_conv = [dec_fc1, dec_fc2], [dec_conv1, dec_conv2]
    return dec_fc, dec_conv

def apply_mnist_encoders(inputs, latent_dim):
    enc_fc1 = tf.layers.dense(inputs, units=500, activation=tf.nn.relu)
    enc_fc2 = tf.layers.dense(enc_fc1, units=200, activation=tf.nn.relu)
    mean_fc3 = tf.layers.dense(enc_fc2, units=latent_dim)
    stddev_fc3 = tf.layers.dense(enc_fc2, units=latent_dim) # log(sigma^2)
    return mean_fc3, stddev_fc3

def apply_mnist_conv_encoders(inputs, latent_dim):
    enc_conv1 = tf.layers.conv2d(inputs, filters=16, kernel_size=(3, 3),
                                 strides=(2, 2), padding='same', activation=tf.nn.relu)
    enc_conv2 = tf.layers.conv2d(enc_conv1, filters=32, kernel_size=(3, 3),
                                 strides=(2, 2), padding='same', activation=tf.nn.relu)
    enc_fc3 = tf.layers.dense(tf.layers.flatten(enc_conv2), units=500, activation=tf.nn.relu)
    enc_fc4 = tf.layers.dense(enc_fc3, units=200, activation=tf.nn.relu)
    mean_fc5 = tf.layers.dense(enc_fc4, units=latent_dim)
    stddev_fc5 = tf.layers.dense(enc_fc4, units=latent_dim) # log(sigma^2)
    return mean_fc5, stddev_fc5

def faces_decoders():
    dec_fc1 = tf.layers.Dense(units=500, activation=tf.nn.relu)
    dec_fc2 = tf.layers.Dense(units=8*8*64, activation=tf.nn.relu)
    dec_conv1 = tf.layers.Conv2DTranspose(filters=128, kernel_size=(3, 3), strides=2,
                                          padding='same', activation=tf.nn.relu)
    dec_conv2 = tf.layers.Conv2DTranspose(filters=64, kernel_size=(3, 3), strides=2,
                                          padding='same', activation=tf.nn.relu)
    dec_conv3 = tf.layers.Conv2DTranspose(filters=1, kernel_size=(3, 3), strides=2,
                                          padding='same', activation=tf.nn.sigmoid)
    dec_fc = [dec_fc1, dec_fc2]
    dec_conv = [dec_conv1, dec_conv2, dec_conv3]
    return dec_fc, dec_conv

def faces_upsample_decoders():
    dec_fc1 = tf.layers.Dense(units=500, activation=tf.nn.relu)
    dec_fc2 = tf.layers.Dense(units=8*8*64, activation=tf.nn.relu)
    dec_upsample1 = lambda x: tf.image.resize_images(
        x, [16, 16], method=tf.image.ResizeMethod.BILINEAR)
    dec_conv1 = tf.layers.Conv2D(filters=128, kernel_size=(3, 3), strides=1,
                                 padding='same', activation=tf.nn.relu)
    dec_conv1_2 = tf.layers.Conv2D(filters=128, kernel_size=(3, 3), strides=1,
                                   padding='same', activation=tf.nn.relu)
    dec_upsample2 = lambda x: tf.image.resize_images(
        x, [32, 32], method=tf.image.ResizeMethod.BILINEAR)
    dec_conv2 = tf.layers.Conv2D(filters=64, kernel_size=(3, 3), strides=1,
                                 padding='same', activation=tf.nn.relu)
    dec_conv2_2 = tf.layers.Conv2D(filters=64, kernel_size=(3, 3), strides=1,
                                   padding='same', activation=tf.nn.relu)
    dec_upsample3 = lambda x: tf.image.resize_images(
        x, [64, 64], method=tf.image.ResizeMethod.BILINEAR)
    dec_conv3 = tf.layers.Conv2D(filters=32, kernel_size=(3, 3), strides=1,
                                 padding='same', activation=tf.nn.relu)
    dec_conv3_2 = tf.layers.Conv2D(filters=1, kernel_size=(3, 3), strides=1,
                                   padding='same', activation=tf.nn.sigmoid)
    dec_fc = [dec_fc1, dec_fc2]
    dec_conv = [dec_upsample1, dec_conv1, dec_conv1_2,
                dec_upsample2, dec_conv2, dec_conv2_2,
                dec_upsample3, dec_conv3, dec_conv3_2]
    return dec_fc, dec_conv

def apply_faces_conv_encoders(inputs):
    enc_conv1 = tf.layers.conv2d(inputs, filters=32, kernel_size=(3, 3),
                                 strides=(2, 2), padding='same', activation=tf.nn.relu)
    enc_conv2 = tf.layers.conv2d(enc_conv1, filters=64, kernel_size=(3, 3),
                                 strides=(2, 2), padding='same', activation=tf.nn.relu)
    enc_conv3 = tf.layers.conv2d(enc_conv2, filters=128, kernel_size=(3, 3),
                                 strides=(2, 2), padding='same', activation=tf.nn.relu)
    return enc_conv3

def apply_faces_fc_encoders(inputs, latent_dim):
    enc_fc3 = tf.layers.dense(inputs, units=500, activation=tf.nn.relu)
    enc_fc4 = tf.layers.dense(enc_fc3, units=200, activation=tf.nn.relu)
    mean_fc5 = tf.layers.dense(enc_fc4, units=latent_dim)
    stddev_fc5 = tf.layers.dense(enc_fc4, units=latent_dim) # log(sigma^2)
    return mean_fc5, stddev_fc5

def apply_faces_encoders(inputs, latent_dim):
    enc_conv3 = apply_faces_conv_encoders(inputs)
    enc_conv3 = tf.layers.flatten(enc_conv3)
    mean_fc5, stddev_fc5 = apply_faces_fc_encoders(enc_conv3, latent_dim)
    return mean_fc5, stddev_fc5

def bernoulli_recon_loss(inputs, reconstructed_inputs):
    eps = 1e-10
    loss = inputs * tf.log(reconstructed_inputs + eps) + \
           (1 - inputs) * tf.log((1 - reconstructed_inputs) + eps)
    return loss

def gaussian_kl_loss(mean, log_stddev_squared):
    loss = 1 + log_stddev_squared - tf.square(mean) - tf.exp(log_stddev_squared)
    return loss


class VAE():
    def __init__(self, vae_fn, input_dim, latent_dim):
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        vae_tensors = vae_fn(input_dim, latent_dim)
        self.inputs = vae_tensors[0]
        self.encoder_mean = vae_tensors[1]
        self.decoder_train_output = vae_tensors[2]
        self.decoder_inputs = vae_tensors[3]
        self.decoder_output = vae_tensors[4]
        self.loss = vae_tensors[5]
        self.trainer = vae_tensors[6]
        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())

    def decode(self, Z):
        decoded_outputs = self.sess.run(self.decoder_output,
                                        {self.decoder_inputs: Z})
        return decoded_outputs

    def encode(self, X):
        # Return the mean from using a Gaussian encoder.
        encoded_inputs = self.sess.run(self.encoder_mean, {self.inputs: X})
        return encoded_inputs

    def train(self, X):
        _, batch_loss = self.sess.run([self.trainer, self.loss],
                                      {self.inputs: X})
        return batch_loss

class ConditionalVAE():
    def __init__(self, vae_fn, input_dim, latent_dim, num_classes):
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.num_classes = num_classes
        vae_tensors = vae_fn(input_dim, latent_dim, num_classes)
        self.inputs = vae_tensors[0]
        self.input_labels = vae_tensors[1]
        self.encoder_mean = vae_tensors[2]
        self.decoder_train_output = vae_tensors[3]
        self.decoder_inputs = vae_tensors[4]
        self.decoder_input_labels = vae_tensors[5]
        self.decoder_output = vae_tensors[6]
        self.loss = vae_tensors[7]
        self.trainer = vae_tensors[8]
        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())

    def decode(self, Z, y):
        decoded_outputs = \
            self.sess.run(self.decoder_output, {self.decoder_inputs: Z,
                                                self.decoder_input_labels: y})
        return decoded_outputs

    def encode(self, X, y):
        # Return the mean from using a Gaussian encoder.
        encoded_inputs = self.sess.run(
            self.encoder_mean, {self.inputs: X, self.input_labels: y})
        return encoded_inputs

    def train(self, X, y):
        _, batch_loss = self.sess.run([self.trainer, self.loss],
                                      {self.inputs: X, self.input_labels: y})
        return batch_loss

def vae(data_dim, latent_dim):
    inputs = tf.placeholder(tf.float32, [None, data_dim])
    batch_size = tf.shape(inputs)[0]
    # Declare decoder components.
    dec_layers = mnist_decoders(data_dim)
    # Encode.
    mean_fc3, stddev_fc3 = apply_mnist_encoders(inputs, latent_dim)
    # Apply reparameterization towards sampling.
    mvn_sampler = tf.contrib.distributions.MultivariateNormalDiag(
        loc=tf.zeros(latent_dim))
    samples = mvn_sampler.sample(batch_size)
    samples = tf.sqrt(tf.exp(stddev_fc3)) * samples + mean_fc3
    # Share parameters between decoder used for training and inference.
    decoder_inputs = tf.placeholder(tf.float32, [None, latent_dim])
    decoder_output = decoder_inputs
    decoder_train_output = samples
    for decoder_layer in dec_layers:
        decoder_output = decoder_layer(decoder_output)
        decoder_train_output = decoder_layer(decoder_train_output)
    # Define loss function.
    recon_loss = -tf.reduce_sum(
        bernoulli_recon_loss(inputs, decoder_train_output), axis=1)
    recon_loss = tf.reduce_mean(recon_loss)
    kl_loss = -0.5 * tf.reduce_sum(
        gaussian_kl_loss(mean_fc3, stddev_fc3), axis=1)
    kl_loss = tf.reduce_mean(kl_loss)
    loss = recon_loss + kl_loss
    trainer = tf.train.AdamOptimizer().minimize(loss)
    return inputs, mean_fc3, decoder_train_output, \
           decoder_inputs, decoder_output, loss, trainer

def vae_mnist_conv(data_dim, latent_dim):
    inputs = tf.placeholder(tf.float32, [None] + list(data_dim))
    batch_size = tf.shape(inputs)[0]
    # Declare decoder components.
    dec_fc, dec_conv = mnist_conv_decoders()
    # Encode.
    mean_fc5, stddev_fc5 = apply_mnist_conv_encoders(inputs, latent_dim)
    # Apply reparameterization towards sampling.
    mvn_sampler = tf.contrib.distributions.MultivariateNormalDiag(
        loc=tf.zeros(latent_dim))
    samples = mvn_sampler.sample(batch_size)
    samples = tf.sqrt(tf.exp(stddev_fc5)) * samples + mean_fc5
    # Share parameters between decoder used for training and inference.
    decoder_inputs = tf.placeholder(tf.float32, [None, latent_dim])
    decoder_output = decoder_inputs
    decoder_train_output = samples
    for decoder_layer in dec_fc:
        decoder_output = decoder_layer(decoder_output)
        decoder_train_output = decoder_layer(decoder_train_output)
    decoder_output = tf.keras.layers.Reshape([7, 7, 32])(decoder_output)
    decoder_train_output = tf.reshape(decoder_train_output, [batch_size, 7, 7, 32])
    for decoder_layer in dec_conv:
        decoder_output = decoder_layer(decoder_output)
        decoder_train_output = decoder_layer(decoder_train_output)
    # Define loss function.
    recon_loss = -tf.reduce_sum(
        bernoulli_recon_loss(inputs, decoder_train_output), axis=(1, 2, 3))
    recon_loss = tf.reduce_mean(recon_loss)
    kl_loss = -0.5 * tf.reduce_sum(
        gaussian_kl_loss(mean_fc5, stddev_fc5), axis=1)
    kl_loss = tf.reduce_mean(kl_loss)
    loss = recon_loss + kl_loss
    trainer = tf.train.AdamOptimizer().minimize(loss)
    return inputs, decoder_train_output, \
           decoder_inputs, decoder_output, loss, trainer

def vae_face_conv(data_dim, latent_dim):
    inputs = tf.placeholder(tf.float32, [None] + list(data_dim))
    batch_size = tf.shape(inputs)[0]
    # Declare decoder components.
    dec_fc, dec_conv = faces_decoders()
    # Encode.
    mean_fc5, stddev_fc5 = apply_faces_encoders(inputs, latent_dim)
    # Apply reparameterization towards sampling.
    mvn_sampler = tf.contrib.distributions.MultivariateNormalDiag(
        loc=tf.zeros(latent_dim))
    samples = mvn_sampler.sample(batch_size)
    samples = tf.sqrt(tf.exp(stddev_fc5)) * samples + mean_fc5
    # Share parameters between decoder used for training and inference.
    decoder_inputs = tf.placeholder(tf.float32, [None, latent_dim])
    decoder_output = decoder_inputs
    decoder_train_output = samples
    for decoder_layer in dec_fc:
        decoder_output = decoder_layer(decoder_output)
        decoder_train_output = decoder_layer(decoder_train_output)
    decoder_output = tf.keras.layers.Reshape([8, 8, 64])(decoder_output)
    decoder_train_output = tf.reshape(decoder_train_output, [batch_size, 8, 8, 64])
    for decoder_layer in dec_conv:
        decoder_output = decoder_layer(decoder_output)
        decoder_train_output = decoder_layer(decoder_train_output)
    # Define loss function.
    recon_loss = -tf.reduce_sum(
        bernoulli_recon_loss(inputs, decoder_train_output), axis=(1, 2, 3))
    recon_loss = tf.reduce_mean(recon_loss)
    kl_loss = -0.5 * tf.reduce_sum(
        gaussian_kl_loss(mean_fc5, stddev_fc5), axis=1)
    kl_loss = tf.reduce_mean(kl_loss)
    loss = recon_loss + kl_loss
    trainer = tf.train.AdamOptimizer().minimize(loss)
    return inputs, mean_fc5, decoder_train_output, \
           decoder_inputs, decoder_output, loss, trainer

def vae_class_conditional(data_dim, latent_dim, labels_dim):
    inputs = tf.placeholder(tf.float32, [None, data_dim])
    input_labels = tf.placeholder(tf.float32, [None, labels_dim])
    batch_size = tf.shape(inputs)[0]
    # Declare decoder components.
    dec_layers = mnist_decoders(data_dim)
    # Encode.
    concat_inputs = tf.concat([inputs, input_labels], axis=1)
    mean_fc3, stddev_fc3 = apply_mnist_encoders(concat_inputs, latent_dim)
    # Apply reparameterization towards sampling.
    mvn_sampler = tf.contrib.distributions.MultivariateNormalDiag(
        loc=tf.zeros(latent_dim))
    samples = mvn_sampler.sample(batch_size)
    samples = tf.sqrt(tf.exp(stddev_fc3)) * samples + mean_fc3
    # Share parameters between decoder used for training and inference.
    decoder_inputs = tf.placeholder(tf.float32, [None, latent_dim])
    decoder_input_labels = tf.placeholder(tf.float32, [None, labels_dim])
    decoder_output = tf.concat([decoder_inputs, decoder_input_labels], axis=1)
    decoder_train_output = tf.concat([samples, input_labels], axis=1)
    for decoder_layer in dec_layers:
        decoder_output = decoder_layer(decoder_output)
        decoder_train_output = decoder_layer(decoder_train_output)
    # Define loss function.
    recon_loss = -tf.reduce_sum(
        bernoulli_recon_loss(inputs, decoder_train_output), axis=1)
    recon_loss = tf.reduce_mean(recon_loss)
    kl_loss = -0.5 * tf.reduce_sum(
        gaussian_kl_loss(mean_fc3, stddev_fc3), axis=1)
    kl_loss = tf.reduce_mean(kl_loss)
    loss = recon_loss + kl_loss
    trainer = tf.train.AdamOptimizer().minimize(loss)
    return inputs, input_labels, mean_fc3, decoder_train_output, \
           decoder_inputs, decoder_input_labels, decoder_output, loss, trainer

def vae_faces_class_conditional(data_dim, latent_dim, labels_dim):
    inputs = tf.placeholder(tf.float32, [None] + list(data_dim))
    input_labels = tf.placeholder(tf.float32, [None, labels_dim])
    batch_size = tf.shape(inputs)[0]
    # Declare decoder components.
    dec_fc, dec_conv = faces_decoders()
    # Encode.
    enc_conv3 = apply_faces_conv_encoders(inputs)
    concat_inputs = tf.concat([tf.layers.flatten(enc_conv3), input_labels], axis=1)
    mean_fc5, stddev_fc5 = apply_faces_fc_encoders(concat_inputs, latent_dim)
    # Apply reparameterization towards sampling.
    mvn_sampler = tf.contrib.distributions.MultivariateNormalDiag(
        loc=tf.zeros(latent_dim))
    samples = mvn_sampler.sample(batch_size)
    samples = tf.sqrt(tf.exp(stddev_fc5)) * samples + mean_fc5
    # Share parameters between decoder used for training and inference.
    decoder_inputs = tf.placeholder(tf.float32, [None, latent_dim])
    decoder_input_labels = tf.placeholder(tf.float32, [None, labels_dim])
    decoder_output = tf.concat([decoder_inputs, decoder_input_labels], axis=1)
    decoder_train_output = tf.concat([samples, input_labels], axis=1)
    for decoder_layer in dec_fc:
        decoder_output = decoder_layer(decoder_output)
        decoder_train_output = decoder_layer(decoder_train_output)
    decoder_output = tf.keras.layers.Reshape([8, 8, 32])(decoder_output)
    decoder_train_output = tf.reshape(decoder_train_output, [batch_size, 8, 8, 32])
    for decoder_layer in dec_conv:
        decoder_output = decoder_layer(decoder_output)
        decoder_train_output = decoder_layer(decoder_train_output)
    # Define loss function.
    recon_loss = -tf.reduce_sum(
        bernoulli_recon_loss(inputs, decoder_train_output), axis=1)
    recon_loss = tf.reduce_mean(recon_loss)
    kl_loss = -0.5 * tf.reduce_sum(
        gaussian_kl_loss(mean_fc5, stddev_fc5), axis=1)
    kl_loss = tf.reduce_mean(kl_loss)
    loss = recon_loss + kl_loss
    trainer = tf.train.AdamOptimizer().minimize(loss)
    return inputs, input_labels, mean_fc5, decoder_train_output, \
           decoder_inputs, decoder_input_labels, decoder_output, loss, trainer

def run():
    data, labels = get_mnist_data()
    data = data.reshape(-1, 784) / 255.
    # data = data.reshape(-1, 28, 28, 1) / 255.
    # data, labels = get_olivetti_data()
    # data = data.reshape(-1, 64, 64, 1)
    # data, labels = get_train_data()
    batch_size = 100
    num_iters = 10000
    img_height, img_width = 28, 28
    # img_height, img_width = 64, 64
    # Declare model.
    latent_dim = 10
    vae_model = VAE(vae, 784, latent_dim)
    # vae_model = VAE(vae_mnist_conv, [28, 28, 1], latent_dim)
    # vae_model = VAE(vae_face_conv, [64, 64, 3], latent_dim)
    # Visualization points.
    num_viz, num_cols = 21, 20
    xs = np.random.multivariate_normal(
        np.zeros(latent_dim), np.eye(latent_dim), num_viz)
    rows = []
    for i in range(num_viz - 1):
        step = (xs[i+1] - xs[i]) / float(num_cols)
        rows.append([xs[i] + step*j for j in range(num_cols)])
    rows = np.array(rows).reshape(-1, latent_dim)
    output_dir = 'out2'
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    # Train.
    for t in range(num_iters):
        idx = np.random.choice(np.arange(len(data)), batch_size, replace=False)
        X_batch = data[idx]
        batch_loss = vae_model.train(X_batch)
        if t % 200 == 0:
            print(t, batch_loss)
            decoded_outputs = vae_model.decode(rows)
            decoded_outputs = decoded_outputs.reshape(num_viz - 1, num_cols, img_height, img_width)
            output_buffer = np.zeros(((num_viz - 1) * img_height, num_cols * img_width))
            for i in range(num_viz - 1):
                for j in range(num_cols):
                    output_buffer[i*img_height:(i+1)*img_height,
                                  j*img_width:(j+1)*img_width] = decoded_outputs[i, j]
            skio.imsave(output_dir + '/%d.png' % t, output_buffer)

def run_class_conditional():
    data, labels = get_mnist_data()
    data = data.reshape(-1, 784) / 255.
    # data, labels = get_olivetti_data()
    # data = data.reshape(-1, 64, 64, 1)
    num_classes = 10
    batch_size = 100
    num_iters = 10000
    img_height, img_width = 28, 28
    # img_height, img_width = 64, 64
    # Declare model.
    latent_dim = 10
    vae_model = ConditionalVAE(vae_class_conditional, 784, latent_dim, num_classes)
    # vae_model = ConditionalVAE(vae_faces_class_conditional, [64, 64, 1], latent_dim, num_classes)
    # Visualization points.
    num_viz, num_cols = 20, 20
    rows, row_labels = [], []
    for i in range(num_viz):
        xs = np.random.multivariate_normal(
            np.zeros(latent_dim), np.eye(latent_dim), 2)
        step = (xs[1] - xs[0]) / float(num_cols)
        rows.append([xs[0] + step*j for j in range(num_cols)])
        row_labels += [i // 2] * num_cols
    rows = np.array(rows).reshape(-1, latent_dim)
    row_labels = np.diag(np.arange(num_classes))[row_labels]
    output_dir = 'out6'
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    # Train.
    for t in range(num_iters):
        idx = np.random.choice(np.arange(len(data)), batch_size, replace=False)
        X_batch, y_batch = data[idx], labels[idx]
        y_batch_one_hot = np.diag(np.arange(num_classes))[y_batch.astype(np.int32)]
        batch_loss = vae_model.train(X_batch, y_batch_one_hot)
        if t % 200 == 0:
            print(t, np.mean(batch_loss))
            decoded_outputs = vae_model.decode(rows, row_labels)
            decoded_outputs = decoded_outputs.reshape(num_viz, num_cols,
                                                      img_height, img_width)
            output_buffer = np.zeros((num_viz * img_height, num_cols * img_width))
            for i in range(num_viz):
                for j in range(num_cols):
                    output_buffer[i*img_height:(i+1)*img_height,
                                  j*img_width:(j+1)*img_width] = decoded_outputs[i, j]
            skio.imsave(output_dir + '/%d.png' % t, output_buffer)

if __name__ == '__main__':
    run()
    # run_class_conditional()
