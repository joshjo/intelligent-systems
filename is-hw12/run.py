import tensorflow as tf
import numpy as np
import cv2 as cv
import plot_utils

import plot_utils
import mnist_data
import vae

IMAGE_SIZE = 28
RESULTS_DIR = 'results'
PRR_n_img_x = 10
PRR_n_img_y = 10
PRR_resize_factor = 1.0


dim_img = IMAGE_SIZE ** 2
learn_rate = 0.001
dim_z = 20
n_hidden = 500
batch_size = 128
add_noise = True
n_epochs = 5


if __name__ == '__main__':
    train_total_data, train_size, _, _, test_data, test_labels = mnist_data.prepare_MNIST_data()

    print('len test', len(test_labels), len(test_data))
    print('test_labels', test_labels[90])

    n_samples = train_size

    x_hat = tf.placeholder(
        tf.float32, shape=[None, dim_img], name='input_img')
    x = tf.placeholder(
        tf.float32, shape=[None, dim_img], name='target_img')
    keep_prob = tf.placeholder(tf.float32, name='keep_prob')
    z_in = tf.placeholder(
        tf.float32, shape=[None, dim_z], name='latent_variable')
    y, z, loss, neg_marginal_likelihood, KL_divergence = vae.autoencoder(
        x_hat, x, dim_img, dim_z, n_hidden, keep_prob)
    train_op = tf.train.AdamOptimizer(learn_rate).minimize(loss)

    PRR = plot_utils.Plot_Reproduce_Performance(
        RESULTS_DIR, PRR_n_img_x, PRR_n_img_y, IMAGE_SIZE, IMAGE_SIZE, PRR_resize_factor)

    x_PRR = test_data[0:PRR.n_tot_imgs, :]

    x_PRR_img = x_PRR.reshape(
        PRR.n_tot_imgs, IMAGE_SIZE, IMAGE_SIZE)
    PRR.save_images(x_PRR_img, name='input.jpg')

    x_PRR = x_PRR * np.random.randint(2, size=x_PRR.shape)
    x_PRR += np.random.randint(2, size=x_PRR.shape)

    x_PRR_img = x_PRR.reshape(PRR.n_tot_imgs, IMAGE_SIZE, IMAGE_SIZE)
    PRR.save_images(x_PRR_img, name='input_noise.jpg')

    # train
    total_batch = int(n_samples / batch_size)
    min_tot_loss = 1e99

    # with tf.Session() as sess:
    sess.run(tf.global_variables_initializer(), feed_dict={keep_prob : 0.9})

    for epoch in range(n_epochs):

        # Random shuffling
        np.random.shuffle(train_total_data)
        train_data_ = train_total_data[:, :-mnist_data.NUM_LABELS]

        # Loop over all batches
        for i in range(total_batch):
            # Compute the offset of the current minibatch in the data.
            offset = (i * batch_size) % (n_samples)
            batch_xs_input = train_data_[offset:(offset + batch_size), :]

            batch_xs_target = batch_xs_input

            # add salt & pepper noise
            if add_noise:
                batch_xs_input = batch_xs_input * np.random.randint(2, size=batch_xs_input.shape)
                batch_xs_input += np.random.randint(2, size=batch_xs_input.shape)
            
            _, tot_loss, loss_likelihood, loss_divergence = sess.run(
                (train_op, loss, neg_marginal_likelihood, KL_divergence),
                feed_dict={x_hat: batch_xs_input, x: batch_xs_target, keep_prob : 0.9})
        
        if min_tot_loss > tot_loss or epoch + 1 == n_epochs:
            min_tot_loss = tot_loss

            # Plot for reproduce performance
            y_PRR = sess.run(y, feed_dict={x_hat: x_PRR, keep_prob : 1})
            y_PRR_img = y_PRR.reshape(PRR.n_tot_imgs, IMAGE_SIZE, IMAGE_SIZE)
            PRR.save_images(y_PRR_img, name="/PRR_epoch_%02d" %(epoch) + ".jpg")

        # print cost every epoch
        print("epoch %d: L_tot %03.2f L_likelihood %03.2f L_divergence %03.2f" % (epoch, tot_loss, loss_likelihood, loss_divergence))
    
    # for i in range(10):
    y_PRR = sess.run(y, feed_dict={x_hat: x_PRR, keep_prob : 1})
    print('y_PRR', y_PRR)
    y_PRR_img = y_PRR.reshape(PRR.n_tot_imgs, IMAGE_SIZE, IMAGE_SIZE)
            # PRR.save_images(y_PRR_img, name="/PRR_epoch_test_%02d" % (i + 1) + ".jpg")
