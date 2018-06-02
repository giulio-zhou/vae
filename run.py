from util import get_mnist_data
import numpy as np
import os
import skimage.io as skio
import tensorflow as tf

def vae(data_dim):
    inputs = tf.placeholder(tf.float32, [None, data_dim])
    batch_size = tf.shape(inputs)[0]
    # Declare decoder components.
    dec_fc1 = tf.layers.Dense(units=200, activation=tf.nn.relu)
    dec_fc2 = tf.layers.Dense(units=500, activation=tf.nn.relu)
    dec_fc3 = tf.layers.Dense(units=data_dim, activation=tf.nn.sigmoid)
    # Encode.
    enc_fc1 = tf.layers.dense(inputs, units=500, activation=tf.nn.relu)
    enc_fc2 = tf.layers.dense(enc_fc1, units=200, activation=tf.nn.relu)
    mean_fc3 = tf.layers.dense(enc_fc2, units=10)
    stddev_fc3 = tf.layers.dense(enc_fc2, units=10) # log(sigma^2)
    # Apply reparameterization towards sampling.
    mvn_sampler = tf.contrib.distributions.MultivariateNormalDiag(
        loc=tf.zeros(10))
    samples = mvn_sampler.sample(batch_size)
    samples = tf.sqrt(tf.exp(stddev_fc3)) * samples + mean_fc3
    # Share parameters between decoder used for training and inference.
    decoder_inputs = tf.placeholder(tf.float32, [None, 10])
    decoder_output = decoder_inputs
    decoder_train_output = samples
    for decoder_layer in [dec_fc1, dec_fc2, dec_fc3]:
        decoder_output = decoder_layer(decoder_output)
        decoder_train_output = decoder_layer(decoder_train_output)
    # Define loss function.
    eps = 1e-10
    recon_loss = -tf.reduce_sum(
        inputs * tf.log(decoder_train_output + eps) + \
        (1 - inputs) * tf.log((1 - decoder_train_output) + eps), axis=1)
    recon_loss = tf.reduce_mean(recon_loss)
    kl_loss = -0.5 * tf.reduce_sum(
        1 + stddev_fc3 - tf.square(mean_fc3) - tf.exp(stddev_fc3), axis=1)
    kl_loss = tf.reduce_mean(kl_loss)
    loss = recon_loss + kl_loss
    trainer = tf.train.AdamOptimizer().minimize(loss)
    return inputs, decoder_train_output, \
           decoder_inputs, decoder_output, loss, trainer

def run():
    data, labels = get_mnist_data()
    data = data.reshape(-1, 784) / 255.
    batch_size = 100
    num_iters = 10000
    # Declare model.
    inputs, decoder_train_output, decoder_inputs, decoder_output, loss, trainer = vae(784)
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    # Visualization points.
    num_viz, num_cols = 21, 20
    xs = np.random.multivariate_normal(np.zeros(10), np.eye(10), num_viz)
    rows = []
    for i in range(num_viz - 1):
        step = (xs[i+1] - xs[i]) / float(num_cols)
        rows.append([xs[i] + step*j for j in range(num_cols)])
    rows = np.array(rows).reshape(-1, 10)
    output_dir = 'out'
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    # Train.
    for t in range(num_iters):
        idx = np.random.choice(np.arange(len(data)), batch_size, replace=False)
        X_batch = data[idx]
        _, batch_loss = sess.run([trainer, loss], {inputs: X_batch})
        if t % 100 == 0:
            print(t, batch_loss)
            decoded_outputs = sess.run(decoder_output,
                                       {decoder_inputs: rows})
            decoded_outputs = decoded_outputs.reshape(num_viz - 1, num_cols, 28, 28)
            output_buffer = np.zeros(((num_viz - 1) * 28, num_cols * 28))
            for i in range(num_viz - 1):
                for j in range(num_cols):
                    output_buffer[i*28:(i+1)*28, j*28:(j+1)*28] = decoded_outputs[i, j]
            skio.imsave(output_dir + '/%d.png' % t, output_buffer)

if __name__ == '__main__':
    run()
