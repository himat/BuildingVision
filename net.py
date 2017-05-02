import tensorflow as tf
import numpy as np
import os
import glob
import argparse

from generator import Generator
from discriminator import conv_net, conv_weights
from util import plot_save_single, plot_save_batch

EPS = 1e-12


parser = argparse.ArgumentParser()
parser.add_argument("--input_dir", required=True, help="base directory name that contains the images")
parser.add_argument("--ckpt_dir", required=True, help="directory to save and restore network variables from")

parser.add_argument("--num_epochs", type=int, default=115, help="how many epochs to run for")
parser.add_argument("--mb_size", type=int, default=9, help="minibatch size")
parser.add_argument("--l1_weight", type=float, default=0.4, help="l1_weight")

parser.add_argument("--mb_to_print", type=int, default=100, help="how often to print in an epoch")
parser.add_argument("--mb_to_save", type=int, default=0, help="how often to save the output during epochs")
parser.add_argument("--epoch_to_save", type=int, default=5, help="how often (in epochs) to save network variables")

parser.add_argument("--sketch_nc", type=int, default=1, help="number of sketch image channels")


OPTIONS = parser.parse_args()

input_dir = OPTIONS.input_dir
train_path = os.path.join(input_dir, "train")

epochs = OPTIONS.num_epochs
mb_size = OPTIONS.mb_size
mb_to_save = OPTIONS.mb_to_save
l1_weight = OPTIONS.l1_weight
epoch_to_save = OPTIONS.epoch_to_save
ckpt_dir = OPTIONS.ckpt_dir
sketch_nc = OPTIONS.sketch_nc # number of sketch image channels

IMAGE_DIM = 128
IMAGE_SIZE = 16384  # 128 x 128
input_nc = 3  # number of input image channels

print("Epochs: ", epochs)
print("Minibatch size: ", mb_size)

# general helper functions
def flatten(l):
    for i in l:
        for j in i:
            yield j


# Discriminator Model
D_W, D_b, D_bn = conv_weights()
theta_D = (list(D_W.values()) + list(D_b.values()) +
           list(flatten(D_bn.values())))


def discriminator(color, sketch, W, b, bn, is_training):
    sketch = tf.reshape(sketch, [-1, 128, 128, sketch_nc])
    color = tf.reshape(color, [-1, 128, 128, 3])
    y = tf.concat([color, sketch], axis=3)
    return conv_net(y, (W, b, bn), is_training=is_training)


# Generator Model
generator = Generator()
theta_G = generator.weights


# Initialize variable saving
saver = tf.train.Saver(max_to_keep=1)
tf.add_to_collection("mb_size", mb_size)
tf.add_to_collection("l1_weight", l1_weight)

dir = os.path.dirname(os.path.realpath(__file__))

filetype = ".jpg"
ground_truth_files_path = os.path.join(dir, train_path, "real")
ground_truth_files = os.path.join(ground_truth_files_path, "*" + filetype)
edges_files_path = os.path.join(dir, train_path, "edges")
edges_files = os.path.join(edges_files_path, "*" + filetype)

truth_filenames_tf = tf.train.match_filenames_once(ground_truth_files)

print("Reading file from: ", ground_truth_files)
truth_filenames_np = glob.glob(ground_truth_files)


np.random.shuffle(truth_filenames_np)
truth_filenames_tf = tf.convert_to_tensor(truth_filenames_np)


def get_edges_file(f):
    # Splits at last occurrence of / to get the file name
    actual_file_name = f.rpartition(os.sep)[2]
    return os.path.join(edges_files_path, actual_file_name)


edges_fnames = [get_edges_file(f) for f in truth_filenames_np]
edges_fnames_tf = tf.convert_to_tensor(edges_fnames)


print("Truth list shape: ", truth_filenames_tf.shape)
print("Edges list shape: ", edges_fnames_tf.shape)
num_train_data = truth_filenames_tf.shape.as_list()[0]

truth_image_name, edges_image_name = tf.train.slice_input_producer(
    [truth_filenames_tf, edges_fnames_tf], shuffle=True)

value_truth_imgfile = tf.read_file(truth_image_name)

# Returns as a tensor for us to use
truth_image = tf.image.decode_jpeg(value_truth_imgfile)
truth_image.set_shape([IMAGE_DIM, IMAGE_DIM, input_nc])
# image = tf.reshape(image, [IMAGE_SIZE*input_nc])
truth_image = tf.cast(truth_image, tf.float32)
truth_image = truth_image/255.0  # Normalize RGB to [0,1]


value_edges_imgfile = tf.read_file(edges_image_name)
edges_image = tf.image.decode_jpeg(value_edges_imgfile)
edges_image.set_shape([IMAGE_DIM, IMAGE_DIM, sketch_nc])
# edges_image = tf.reshape(edges_image, [IMAGE_DIM, IMAGE_DIM])
# edges_image = edges_image.squeeze() # Get rid of single channel third dim
edges_image = tf.cast(edges_image, tf.float32)
edges_image = edges_image/255.0

min_queue_examples = epochs*mb_size
num_threads = 4
# Background thread to batch images
# [truth_images_batch, edges_images_batch] = tf.train.shuffle_batch(
    # [truth_image, edges_image],  # image_tensor
    # batch_size=mb_size,
    # capacity=min_queue_examples + num_threads*mb_size,
    # min_after_dequeue=min_queue_examples,
    # # shapes=([IMAGE_DIM, IMAGE_DIM, input_nc]),
    # num_threads=num_threads,
    # allow_smaller_final_batch=True)

[truth_images_batch, edges_images_batch] = tf.train.batch(
    [truth_image, edges_image],
    batch_size=mb_size,
	capacity=30,
	num_threads=mb_size)

print("Batch shape ", truth_images_batch.shape)

X_sketch = tf.placeholder(
    tf.float32, shape=[mb_size, IMAGE_DIM, IMAGE_DIM, 1], name='X_sketch')
X_prog_sketch = tf.placeholder(
    tf.float32, shape=[1, IMAGE_DIM, IMAGE_DIM, 1], name='X_prog_sketch')
tf.add_to_collection("X_sketch", X_sketch)
X_ground_truth = tf.placeholder(
    tf.float32, shape=[mb_size, IMAGE_DIM,
                       IMAGE_DIM, input_nc], name='X_ground_truth')
X_is_training = tf.placeholder(tf.bool, shape=[], name='X_is_training')

# Generate CGAN outputs
G_sample = generator(X_sketch)
G_test = generator(X_sketch, is_training=False)
G_prog_test = tf.reshape(generator(X_prog_sketch, is_training=False),
    [IMAGE_DIM, IMAGE_DIM, 3])
tf.add_to_collection("G_test", G_test)
D_real, D_logit_real = discriminator(X_ground_truth, X_sketch, D_W, D_b, D_bn,
                                     X_is_training)
D_fake, D_logit_fake = discriminator(G_sample, X_sketch, D_W, D_b, D_bn,
                                     X_is_training)

# Calculate CGAN (classic) losses
D_loss = tf.reduce_mean(-(tf.log(D_real + EPS) + tf.log(1. - D_fake + EPS)))
G_L1_loss = tf.reduce_mean(tf.abs(X_ground_truth - G_sample))
G_loss = tf.reduce_mean(-tf.log(D_fake + EPS)) + G_L1_loss*l1_weight

# Calculate CGAN (alternative) losses
# D_loss_real = tf.reduce_mean(
    # tf.nn.sigmoid_cross_entropy_with_logits(
        # logits=D_logit_real, labels=tf.ones_like(D_logit_real)))
# D_loss_fake = tf.reduce_mean(
    # tf.nn.sigmoid_cross_entropy_with_logits(
        # logits=D_logit_fake, labels=tf.zeros_like(D_logit_fake)))

# D_loss = D_loss_real + D_loss_fake
# lmbda = 1  # fix scaling
# G_loss = tf.reduce_mean(
    # tf.nn.sigmoid_cross_entropy_with_logits(
        # logits=D_logit_fake,
        # labels=tf.ones_like(D_logit_fake))) #+ lmbda*tf.reduce_mean(
            # #X_ground_truth - G_sample)

# Apply an optimizer to minimize the above loss functions
D_solver = tf.train.AdamOptimizer().minimize(D_loss, var_list=theta_D)
G_solver = tf.train.AdamOptimizer().minimize(G_loss, var_list=theta_G)

if not os.path.exists('out/'):
    os.makedirs('out/')
if not os.path.exists(ckpt_dir):
    os.makedirs(ckpt_dir)

epoch_to_print = 1
mb_to_print = OPTIONS.mb_to_print

# Demonstrate Model Progression
set_file_name = os.path.join(edges_files_path, "00000001_016.jpg")
set_edge_imgfile = tf.read_file(set_file_name)
set_edge_image = tf.image.decode_jpeg(set_edge_imgfile)
set_edge_image.set_shape([IMAGE_DIM, IMAGE_DIM, sketch_nc])
set_edge_image = tf.cast(set_edge_image, tf.float32)
set_edge_image = set_edge_image/255.0
set_edge_image = tf.reshape(set_edge_image, [1, IMAGE_DIM, IMAGE_DIM, sketch_nc])

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    
    # Starts background threads for image reading
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord)

    set_edge_image = sess.run(set_edge_image)

    for i in range(epochs):

        if i % epoch_to_print == 0:
            print("Epoch ", i)

        for mb_idx in range(num_train_data // mb_size):

            # Get next batch
            [X_truth_batch, X_edges_batch] = sess.run([truth_images_batch,
                                                       edges_images_batch])

            # print(sess.run((D_fake, D_logit_fake)))

            # for j in range(3):
            _, D_loss_curr = sess.run([D_solver, D_loss],
                                      feed_dict={X_ground_truth: X_truth_batch,
                                                 X_sketch: X_edges_batch,
                                                 X_is_training: True})
            _, G_loss_curr = sess.run([G_solver, G_loss],
                                      feed_dict={X_ground_truth: X_truth_batch,
                                                 X_sketch: X_edges_batch,
                                                 X_is_training: True})

            if mb_idx % mb_to_print == 0:
                print("Batch ", mb_idx)
                print("D loss: {:.8}".format(D_loss_curr))
                print("G loss: {:.8}".format(G_loss_curr))

            if (not mb_to_save == 0) and mb_idx % mb_to_save == 0:
                produced_image = sess.run(G_test,
                                      feed_dict={X_sketch: X_edges_batch})
                plot_save_batch(produced_image[0:4], mb_idx, save_only=True,
                                prefix=(str(i)+"e"))

        if i % epoch_to_print == 0:
            produced_image = sess.run(G_test,
                                  feed_dict={X_sketch: X_edges_batch})
            produced_set_image = sess.run(G_prog_test,
                                  feed_dict={X_prog_sketch: set_edge_image})

            plot_save_batch(produced_image[0:4], i, save_only=True)
            plot_save_single(produced_set_image, save_only=True, 
                                          dir="prog", name="cat%d" % i)

            print("D loss: {:.4}".format(D_loss_curr))
            print("G loss: {:.4}".format(G_loss_curr))

            print()

        if i % epoch_to_save == 0:
            saver.save(sess, os.path.join(ckpt_dir, "model"), global_step=i)

    # Stops background threads
    coord.request_stop()
    coord.join(threads)
