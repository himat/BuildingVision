import tensorflow as tf
import numpy as np
import os
import glob
import argparse

from util import plot_save_single, plot_save_batch

parser = argparse.ArgumentParser()
parser.add_argument("--input_dir", required=True, help="base directory name that contains the images")
parser.add_argument("--ckpt_dir", required=True, help="directory to save and restore network variables from")
parser.add_argument("--test_save_dir", default="test-out", help="dir to save produced images in test mode")
parser.add_argument("--mb_size", type=int, default=4, help="minibatch size")
parser.add_argument("--sketch_nc", type=int, default=1, help="number of sketch image channels")

OPTIONS = parser.parse_args()
input_dir = OPTIONS.input_dir
ckpt_dir = OPTIONS.ckpt_dir
test_save_dir = OPTIONS.test_save_dir
mb_size = OPTIONS.mb_size
sketch_nc = OPTIONS.sketch_nc # number of sketch image channels

test_path = os.path.join(input_dir, "test")

IMAGE_DIM = 128
input_nc = 3  # number of input image channels
        
if not os.path.exists(test_save_dir):
    os.makedirs(test_save_dir)

dir = os.path.dirname(os.path.realpath(__file__))

filetype = ".jpg"
edges_files_path = os.path.join(dir, test_path, "edges")
edges_files = os.path.join(edges_files_path, "*" + filetype)
print("Reading file from: ", edges_files)
edges_filenames_np = glob.glob(edges_files)
edges_filenames_tf = tf.convert_to_tensor(edges_filenames_np)
print("Edges list shape: ", edges_filenames_tf.shape)
num_data = edges_filenames_tf.shape.as_list()[0]

out_fnames = [f.rpartition(os.sep)[2] for f in edges_filenames_np]
out_fnames_tf = tf.convert_to_tensor(out_fnames)

edges_image_name, out_image_name = tf.train.slice_input_producer(
    [edges_filenames_tf, out_fnames_tf])
value_edges_imgfile = tf.read_file(edges_image_name)
edges_image = tf.image.decode_jpeg(value_edges_imgfile)
edges_image.set_shape([IMAGE_DIM, IMAGE_DIM, sketch_nc])
edges_image = tf.cast(edges_image, tf.float32)
edges_image = edges_image/255.0

[edges_images_batch, out_images_batch] = tf.train.batch(
    [edges_image, out_image_name],
    batch_size=mb_size,
    capacity=30,
    num_threads=mb_size)
print("Batch shape ", edges_images_batch.shape)

with tf.Session() as sess:
    meta_path = tf.train.get_checkpoint_state(ckpt_dir).model_checkpoint_path
    saver = tf.train.import_meta_graph(meta_path + ".meta")
    saver.restore(sess, tf.train.latest_checkpoint(ckpt_dir))

    print("Training minibatch size: ", tf.get_collection("mb_size")[0])
    print("Training L1 weight: ", tf.get_collection("l1_weight")[0])

    # Starts background threads for image reading
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord)

    X_sketch = tf.get_collection("X_sketch")[0]
    G_test = tf.get_collection("G_test")[0]

    for batch_idx in range(num_data // mb_size):
        print("Batch ", batch_idx)
        [X_edges_batch, X_out_batch] = sess.run([edges_images_batch, out_images_batch])
        produced_image_batch = sess.run(G_test,
                              feed_dict={X_sketch: X_edges_batch})
        for i, img in enumerate(produced_image_batch):
            plot_save_single(img, save_only=True, dir=test_save_dir, name=X_out_batch[i].decode())

    # Stops background threads
    coord.request_stop()
    coord.join(threads)
