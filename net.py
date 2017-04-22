import tensorflow as tf
import numpy as np

from generator import Generator
from discriminator import conv_net


G = Generator()
generator = G.eval


def discriminator(x, g):
    x = tf.reshape(x, [-1, 128, 128, 1])
    g = tf.reshape(g, [-1, 128, 128, 3])
    y = tf.concat([x, g], 3)
    return conv_net(y)


def next_data_batch(minibatch_size):
    pass

# --> Add conditional stuff
G_sample = generator(Z)
D_real = discriminator(X)
D_fake = discriminator(G_sample)

D_loss = -tf.reduce_mean(tf.log(D_real) + tf.log(1. - D_fake))
G_loss = -tf.reduce_mean(tf.log(D_fake)) + tf.reduce_mean(X - D_fake)

# Apply an optimizer here to minimize the above loss functions
D_solver = tf.train.AdamOptimizer().minimize(D_loss) # --> add var_list
G_solver = tf.train.AdamOptimizer().minimize(G_loss) # --> add var_list

sess = tf.Session()
sess.run(tf.global_variables_initializer())

epochs = 15
minib_size = 4

for i in range(epochs):
    print("Epoch %d" % i)

    X_batch = next_data_batch(minib_size)  # TODO: Change this to something

    _, D_loss_curr = sess.run([D_solver, D_loss],
                              feed_dict={X: X_batch, Z: noise_Z(minib_size)})
    _, G_loss_curr = sess.run([G_solver, G_loss],
                              feed_dict={Z: noise_Z(minib_size)})

