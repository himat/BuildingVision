import tensorflow as tf
import numpy as np


from generator import u_net
from discriminator import conv_net

minib_size = 128
X_dim = 128*128
y_dim = 128*128
Z_dim = 100

X = tf.placeholder(tf.float32, shape=[None, X_dim])
y = tf.placeholder(tf.float32, shape=[None, y_dim])
Z = tf.placeholder(tf.float32, shape=[None, Z_dim])

def generator(x):
    return u_net(x)


def discriminator(x, g):
    x = tf.reshape(x, [-1, 128, 128, 1])
    g = tf.reshape(g, [-1, 128, 128, 3])
    y = tf.concat([x, g], 3)
    return conv_net(y)


def next_data_batch(minibatch_size):
    pass

# --> Add conditional stuff
G_sample = generator(Z) # add conditional parameter
D_real = discriminator(y,X)
D_fake = discriminator(y,G_sample)

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

