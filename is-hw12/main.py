import tensorflow as tf

from model import Model


if __name__ == '__main__':

    keep_prob = tf.placeholder(tf.float32, name='keep_prob')
    m = Model(200, keep_prob)
    sess = tf.Session()
    sess.run(tf.global_variables_initializer(), feed_dict={keep_prob : 0.9})

    m.train(sess)
    img = m.predict(sess)

    print('img', img)

    sess.close()
