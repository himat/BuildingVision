import tensorflow as tf
import numpy as np
import os
import matplotlib.pyplot as plt

from generator import u_net
from discriminator import conv_net, conv_weights

epochs = 10
minib_size = 14

IMAGE_DIM = 128
IMAGE_SIZE = 16384 # 128 x 128
input_nc = 3 # number of input image channels

train_path = "/sample_data" #"data/"
test_path = "data/**"

def generator(x):
    return u_net(x)

def discriminator(x, g, W, b):
    x = tf.reshape(x, [-1, 128, 128, 1])
    g = tf.reshape(g, [-1, 128, 128, 3])
    y = tf.concat([x, g], 3)
    return conv_net(y, (W, b))

def next_data_batch(minibatch_size):
    pass

dir = os.path.dirname(os.path.realpath(__file__))
input_folder = dir + train_path + "/*.jpg"

filenames = tf.train.match_filenames_once(input_folder)
filename_queue = tf.train.string_input_producer(filenames)

image_reader = tf.WholeFileReader()
key_fname, value_imgfile = image_reader.read(filename_queue)
print "file: ", key_fname

# Returns as a tensor for us to use 
image = tf.image.decode_jpeg(value_imgfile)
# image = tf.reshape(image, [IMAGE_SIZE*input_nc])
image = tf.cast(image, tf.float32)
image = image/255.0 # Normalize RGB to [0,1]
print "image: ", type(image)

min_queue_examples = mb_size 

# Background thread to batch images
images_batch = tf.train.shuffle_batch(
                [image],  #image_tensor
                batch_size=mb_size,
                capacity=min_queue_examples + 2*mb_size,
                min_after_dequeue=min_queue_examples,
                shapes=([IMAGE_DIM, IMAGE_DIM, input_nc]),
                num_threads=4,
                allow_smaller_final_batch=True)


X_sketch = tf.placeholder(tf.float32, shape=[None, IMAGE_SIZE*input_nc], name='X')
# Z = tf.placeholder(tf.float32, shape=[None, 100])
X_ground_truth = tf.placeholder(tf.float32, shape=[None, IMAGE_SIZE*input_nc], name='X_ground_truth')

# --> Add conditional stuff
G_sample = generator(X_sketch) # add conditional parameter

D_W, D_b = conv_weights()
D_theta.extend(D_W.values())
D_theta.extend(D_b.values())

D_real, D_logit_real = discriminator(X_ground_truth, X_sketch, (D_W, D_b))
D_fake, D_logit_fake = discriminator(X_ground_truth, G_sample, (D_W, D_b))

# Classic GAN Loss
# D_loss = -tf.reduce_mean(tf.log(D_real) + tf.log(1. - D_fake))
# G_loss = -tf.reduce_mean(tf.log(D_fake)) + tf.reduce_mean(X_sketch - D_fake)

D_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=D_logit_real, 
    labels=tf.ones_like(D_logit_real)))
D_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=D_logit_fake, 
    labels=tf.zeros_like(D_logit_fake)))
D_loss = D_loss_real + D_loss_fake
G_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=D_logit_fake, 
    labels=tf.ones_like(D_logit_fake))) + tf.reduce_mean(X_sketch - D_fake)


(D_W, D_b) = conv_weights()
theta_D = list(D_W.values) + list(D_b.values)
theta_G = [] ### FILL THIS IN

# Apply an optimizer here to minimize the above loss functions
D_solver = tf.train.AdamOptimizer().minimize(D_loss, var_list = theta_D)
G_solver = tf.train.AdamOptimizer().minimize(G_loss, var_list = theta_G)


with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    # Starts background threads for image reading
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord)

    for i in range(epochs):

	if i % 1000 == 0:
	    print("Epoch ", i)

        # Get next batch
        X_sketches = images_batch.eval()

        X_ground_truth = 0 ## Figure out how to get sketches 

	_, D_loss_curr = sess.run([D_solver, D_loss],
                feed_dict={X_sketch: X_sketches, X_ground_truth: X_true})
	_, G_loss_curr = sess.run([G_solver, G_loss],
                feed_dict={X_ground_truth: X_true})

    # Stops background threads
    coord.request_stop()
    coord.join(threads)

