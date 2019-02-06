import matplotlib as mpl
mpl.use('TkAgg')
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import cv2 as cv
import plot_utils

from collections import defaultdict

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
n_epochs = 1


def get_label(arr):
    i = -1
    for i, x in enumerate(arr):
        if x == 1:
            break
    return i


class Model(object):
    def __init__(self, num_epochs, keep_prob):
        self.num_epochs = num_epochs
        self.keep_prob = keep_prob
        self.train_total_data, train_size, _, _, test_data, test_labels = mnist_data.prepare_MNIST_data()
        map_test_labels = {}

        map_labels = defaultdict(list)

        for i in range(100):
            item = test_labels[i]
            map_labels[get_label(item)].append(i)

        self.n_samples = train_size

        self.x_hat = tf.placeholder(
            tf.float32, shape=[None, dim_img], name='input_img')
        self.x = tf.placeholder(
            tf.float32, shape=[None, dim_img], name='target_img')
        # self.keep_prob = tf.placeholder(tf.float32, name='keep_prob')
        z_in = tf.placeholder(
            tf.float32, shape=[None, dim_z], name='latent_variable')
        self.y, self.z, self.loss, self.neg_marginal_likelihood, self.KL_divergence = vae.autoencoder(
            self.x_hat, self.x, dim_img, dim_z, n_hidden, self.keep_prob)
        self.train_op = tf.train.AdamOptimizer(learn_rate).minimize(self.loss)

        self.PRR = plot_utils.Plot_Reproduce_Performance(
            RESULTS_DIR, PRR_n_img_x, PRR_n_img_y, IMAGE_SIZE, IMAGE_SIZE, PRR_resize_factor)

        self.x_PRR = test_data[0:self.PRR.n_tot_imgs, :]

        x_PRR_img = self.x_PRR.reshape(
            self.PRR.n_tot_imgs, IMAGE_SIZE, IMAGE_SIZE)
        self.PRR.save_images(x_PRR_img, name='input.jpg')

        self.x_PRR = self.x_PRR * np.random.randint(2, size=self.x_PRR.shape)
        self.x_PRR += np.random.randint(2, size=self.x_PRR.shape)

        x_PRR_img = self.x_PRR.reshape(self.PRR.n_tot_imgs, IMAGE_SIZE, IMAGE_SIZE)
        self.PRR.save_images(x_PRR_img, name='input_noise.jpg')

        # train
        self.total_batch = int(self.n_samples / batch_size)


    def train(self, sess):
        PRR = self.PRR
        keep_prob = self.keep_prob
        train_total_data = self.train_total_data
        n_samples = self.n_samples
        total_batch = self.total_batch
        train_op = self.train_op
        loss = self.loss
        x = self.x
        y = self.y
        z = self.z
        x_hat = self.x_hat
        neg_marginal_likelihood = self.neg_marginal_likelihood
        x_PRR = self.x_PRR
        KL_divergence = self.KL_divergence

        min_tot_loss = 1e99

        for epoch in range(self.num_epochs):

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
            print("epoch %d: L_tot %03.2f L_likelihood %03.2f L_divergence %03.2f" % (
                epoch, tot_loss, loss_likelihood, loss_divergence))

    def predict(self, sess):
        x_PRR = self.x_PRR
        y = self.y
        x_hat = self.x_hat
        PRR = self.PRR
        keep_prob = self.keep_prob
        # for i in range(10):
        y_PRR = sess.run(y, feed_dict={x_hat: x_PRR, keep_prob : 1})
        print('y_PRR', y_PRR)
        y_PRR_img = y_PRR.reshape(PRR.n_tot_imgs, IMAGE_SIZE, IMAGE_SIZE)
        PRR.save_images(y_PRR_img, name="/predict.jpg")
        img = cv.imread('results/predict', 0)
        return img
