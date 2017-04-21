import tensorflow as tf
from tensorflow.contrib.layers import batch_norm


def lrelu(x, alpha=0.2):
    return tf.maximum(x * alpha, x)


def conv(x, W, b, strides=2, decay=0.99):
    x = tf.nn.conv2d(x, W, strides=[1, strides, strides, 1], padding='SAME')
    x = tf.nn.bias_add(x, b)
    x = batch_norm(x, decay=decay, is_training=True,
                   updates_collections=None, reuse=True)
    return lrelu(x, alpha=0.2)


def maxpool(x, k=2):
    return tf.nn.max_pool(x, ksize=[1, k, k, 1], strides=[1, k, k, 1],
                          padding='SAME')


def dense(x, W, b):
    x = tf.matmul(x, W)
    x = tf.add(x, b)
    return tf.nn.sigmoid(x)


def filters(input, output, shape=(4, 4)):
    (s1, s2) = shape
    return tf.Variable(tf.random_normal([s1, s2, input, output]))


def bias(shape):
    return tf.Variable(tf.random_normal([shape]))


def denseW(input, output):
    return tf.Variable(tf.random_normal([input, output]))


def conv_net(x):
    weights = {
        'c1': filters(1, 64),
        'c2': filters(64, 128),
        'c3': filters(64, 256),
        'c4': filters(256, 512),
        'c5': filters(512, 512),
        'c6': filters(512, 512),
        'd1': denseW(5*5*512, 1),  # TODO check this dimension
    }

    biases = {
        'c1': bias(64),
        'c2': bias(128),
        'c3': bias(256),
        'c4': bias(512),
        'c5': bias(512),
        'c6': bias(512),
        'd1': bias(1),
    }

    x = tf.reshape(x, shape=[-1, 128, 128, 4])

    c1 = conv(x, weights['c1'], biases['c1'], strides=2)

    c2 = conv(c1, weights['c2'], biases['c2'], strides=2)

    c3 = conv(c2, weights['c3'], biases['c3'], strides=2)

    c4 = conv(c3, weights['c4'], biases['c4'], strides=2)

    c5 = conv(c4, weights['c5'], biases['c5'], strides=2)

    c6 = conv(c5, weights['c6'], biases['c6'], strides=2)

    d1 = dense(c6, weights['d1'], biases['d1'])

    return d1


def test_discriminator():
    pass


if __name__ == "__main__":
    test_discriminator()
