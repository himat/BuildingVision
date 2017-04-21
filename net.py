import tensorflow as tf
import numpy as np

from generator import u_net


def generator(x):
    return u_net(x)


def discriminator(x):
    return discriminator.conv_net(x)


def next_data_batch(minibatch_size):
    pass


D_loss = 0
G_loss = 0

# Apply an optimizer here to minimize the above loss functions
D_solver = 0
G_solver = 0

sess = tf.Session()
sess.run(tf.global_variables_initializer())

epochs = 10000
minib_size = 128

for i in range(epochs):

    if i % 1000 == 0:
        print("Epoch ", i)

    X_batch = next_data_batch(minib_size)  # TODO: Change this to something

    _, D_loss_curr = sess.run([D_solver, D_loss],
                              feed_dict={X: X_batch, Z: noise_Z(minib_size)})
    _, G_loss_curr = sess.run([G_solver, G_loss],
                              feed_dict={Z: noise_Z(minib_size)})

