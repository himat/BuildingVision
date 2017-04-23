import tensorflow as tf
from tensorflow.contrib.layers import batch_norm


def lrelu(x, alpha=0.2):
    return tf.maximum(x * alpha, x)


def conv(x, W, b, strides=2, decay=0.99):
    x = tf.nn.conv2d(x, W, strides=[1, strides, strides, 1], padding='SAME')
    x = tf.nn.bias_add(x, b)
    x = batch_norm(x, decay=decay, is_training=True,
                   updates_collections=None)
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
    return tf.Variable(tf.random_normal([s1, s2, input, output], stddev=0.02))


def bias(shape):
    return tf.Variable(tf.random_normal([shape], stddev=0.02))


def denseW(input, output):
    return tf.Variable(tf.random_normal([input, output], stddev=0.02))


def conv_weights():
    weights = {
        'c1': filters(4, 64),
        'c2': filters(64, 128),
        'c3': filters(128, 256),
        'c4': filters(256, 512),
        'c5': filters(512, 512),
        'c6': filters(512, 512),
        'c7': filters(512, 1),
    }

    biases = {
        'c1': bias(64),
        'c2': bias(128),
        'c3': bias(256),
        'c4': bias(512),
        'c5': bias(512),
        'c6': bias(512),
        'c7': bias(1),
    }
    return (weights, biases)


def conv_net(x, vars):
    weights, biases = vars
    x = tf.reshape(x, shape=[-1, 128, 128, 4])

    c1 = conv(x, weights['c1'], biases['c1'], strides=2)

    c2 = conv(c1, weights['c2'], biases['c2'], strides=2)

    c3 = conv(c2, weights['c3'], biases['c3'], strides=2)

    c4 = conv(c3, weights['c4'], biases['c4'], strides=2)

    c5 = conv(c4, weights['c5'], biases['c5'], strides=2)

    c6 = conv(c5, weights['c6'], biases['c6'], strides=2)

    c7 = conv(c6, weights['c7'], biases['c7'], strides=2)

    res = tf.reshape(c7, [-1, 1])
    return (tf.nn.sigmoid(res), res)


def test_conv_net():
    x = tf.Variable(tf.random_normal([214, 128, 128, 4]))
    pred = conv_net(x)
    print(pred.get_shape())


def test_discriminator():
    from tensorflow.examples.tutorials.mnist import input_data

    mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
    n_input = 28 * 28
    n_classes = 10
    x = tf.placeholder(tf.float32, [None, n_input])
    y = tf.placeholder(tf.float32, [None, n_classes])

    weights = {
        'c1': filters(1, 32),
        'c2': filters(32, 128),
        'c3': filters(128, 256),
        'd1': denseW(7*7*256, 1024),  # TODO check this dimension
        'd2': denseW(1024, 10),
    }

    biases = {
        'c1': bias(32),
        'c2': bias(128),
        'c3': bias(256),
        'd1': bias(1024),
        'd2': bias(10),
    }

    x = tf.reshape(x, shape=[-1, 27, 27, 1])

    c1 = conv(x, weights['c1'], biases['c1'], strides=2)

    c2 = conv(c1, weights['c2'], biases['c2'], strides=2)

    c3 = conv(c2, weights['c3'], biases['c3'], strides=2)

    d1 = tf.reshape(c3, [-1, weights['d1'].get_shape().as_list()[0]])
    d1 = dense(d1, weights['d1'], biases['d1'])
    d2 = dense(d1, weights['d2'], biases['d2'])

    pred = d2
    cost = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y))
    optimizer = tf.train.AdamOptimizer(learning_rate=0.001).minimize(cost)
    correct_pred = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
