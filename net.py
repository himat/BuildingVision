import tensorflow as tf
import numpy as np


from generator import u_net
from discriminator import conv_net, conv_weights

minib_size = 128
X_dim = 128*128
y_dim = 128*128
Z_dim = 100

X = tf.placeholder(tf.float32, shape=[None, X_dim])
y = tf.placeholder(tf.float32, shape=[None, y_dim])
Z = tf.placeholder(tf.float32, shape=[None, Z_dim])

""" Discriminator """
D_theta = []

""" Generator """
G_theta = []


def generator(x):
    return u_net(x)


def discriminator(x, g, W, b):
    x = tf.reshape(x, [-1, 128, 128, 1])
    g = tf.reshape(g, [-1, 128, 128, 3])
    y = tf.concat([x, g], 3)
    return conv_net(y, (W, b))


def next_data_batch(minibatch_size):
    pass

# --> Add conditional stuff


G_sample = generator(Z)  # add conditional parameter

D_W, D_b = conv_weights()
D_theta.extend(D_W.values())
D_theta.extend(D_b.values())

D_real, D_real_res = discriminator(y, X, (D_W, D_b))
D_fake, D_fake_res = discriminator(y, G_sample, (D_W, D_b))

D_loss = -tf.reduce_mean(tf.log(D_real) + tf.log(1. - D_fake))
G_loss = -tf.reduce_mean(tf.log(D_fake)) + tf.reduce_mean(X - D_fake)

# Apply an optimizer here to minimize the above loss functions
D_solver = tf.train.AdamOptimizer().minimize(D_loss, var_list = D_theta)
G_solver = tf.train.AdamOptimizer().minimize(G_loss, var_list = G_theta)

sess = tf.Session()
sess.run(tf.global_variables_initializer())

epochs = 10000

for i in range(epochs):

    if i % 1000 == 0:
        print("Epoch ", i)

    X_batch = next_data_batch(minib_size)  # TODO: Change this to something

    y_batch = next_data_batch(minib_size) # CHANGE TO GET OUTLINES

    _, D_loss_curr = sess.run([D_solver, D_loss],
                              feed_dict={X: X_batch, Z: noise_Z(minib_size),
                              y: y_batch})
    _, G_loss_curr = sess.run([G_solver, G_loss],
                              feed_dict={Z: noise_Z(minib_size), y: y_batch})

