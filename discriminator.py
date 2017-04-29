import tensorflow as tf
from tensorflow.contrib.layers import batch_norm


def lrelu(x, a=0.2):
    with tf.name_scope("lrelu"):
        # adding these together creates the leak part and linear part
        # then cancels them out by subtracting/adding an absolute value term
        # leak: a*x/2 - a*abs(x)/2
        # linear: x/2 + abs(x)/2

        # this block looks like it has 2 inputs on the graph unless we do this
        x = tf.identity(x)
        return (0.5 * (1 + a)) * x + (0.5 * (1 - a)) * tf.abs(x)


def batchnorm(input):
    with tf.variable_scope("batchnorm"):
        # this block looks like it has 3 inputs on the graph unless we do this
        input = tf.identity(input)

        channels = input.get_shape()[3]
        offset = tf.get_variable("offset", [channels],
                                 dtype=tf.float32,
                                 initializer=tf.zeros_initializer())
        scale = tf.get_variable("scale", [channels],
                                dtype=tf.float32,
                                initializer=tf.random_normal_initializer(1.0,
                                                                         0.02))
        mean, variance = tf.nn.moments(input, axes=[0, 1, 2], keep_dims=False)
        variance_e = 1e-5
        normalized = tf.nn.batch_normalization(input, mean,
                                               variance, offset, scale,
                                               variance_epsilon=variance_e)
        return normalized


def conv(x, W, b, strides=2, decay=0.99, is_training=True):
    x = tf.nn.conv2d(x, W, strides=[1, strides, strides, 1], padding='SAME')
    x = tf.nn.bias_add(x, b)
    x = lrelu(x, a=0.2)
    mean, variance = tf.nn.moments(x, axes=[0, 1, 2], keep_dims=False)
    x = tf.nn.batch_normalization(x, mean, variance, None, None,
                                  variance_epsilon=1e-5)
    # x = batch_norm(x, decay=decay, is_training=is_training,
    #                updates_collections=None)
    # x = batchnorm(x)
    return x


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


def conv_net(x, vars, is_training=True):
    weights, biases = vars
    x = tf.reshape(x, shape=[-1, 128, 128, 4])

    c1 = conv(x, weights['c1'], biases['c1'],
              is_training=is_training, strides=2)

    c2 = conv(c1, weights['c2'], biases['c2'],
              is_training=is_training, strides=2)

    c3 = conv(c2, weights['c3'], biases['c3'],
              is_training=is_training, strides=2)

    c4 = conv(c3, weights['c4'], biases['c4'],
              is_training=is_training, strides=2)

    c5 = conv(c4, weights['c5'], biases['c5'],
              is_training=is_training, strides=2)

    c6 = conv(c5, weights['c6'], biases['c6'],
              is_training=is_training, strides=2)

    c7 = conv(c6, weights['c7'], biases['c7'],
              is_training=is_training, strides=2)

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

    weights = {
        'c1': filters(1, 32),
        'c2': filters(32, 128),
        'c3': filters(128, 256),
        'd1': denseW(4*4*256, 128),  # TODO check this dimension
        'd2': denseW(128, 10),
    }

    biases = {
        'c1': bias(32),
        'c2': bias(128),
        'c3': bias(256),
        'd1': bias(128),
        'd2': bias(10),
    }

    x = tf.placeholder(tf.float32, [None, 784], name='x')
    y = tf.placeholder(tf.float32, [None, n_classes], name='y')

    v = tf.reshape(x, shape=[-1, 28, 28, 1])

    c1 = conv(v, weights['c1'], biases['c1'], strides=2)

    c2 = conv(c1, weights['c2'], biases['c2'], strides=2)

    c3 = conv(c2, weights['c3'], biases['c3'], strides=2)

    shape = weights['d1'].get_shape().as_list()[0]
    print(shape)
    d1 = tf.reshape(c3, [-1, shape])
    d1 = dense(d1, weights['d1'], biases['d1'])
    d2 = dense(d1, weights['d2'], biases['d2'])

    pred = d2
    cost = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y))
    optimizer = tf.train.AdamOptimizer(learning_rate=0.001).minimize(cost)
    correct_pred = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

    batch_size = 200
    hm_epochs = 2

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        def test():
            correct = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
            accuracy = tf.reduce_mean(tf.cast(correct, 'float'))
            print('Accuracy:', accuracy.eval({x: mnist.test.images[:256],
                                              y: mnist.test.labels[:256]}))

        for epoch in range(hm_epochs):
            epoch_loss = 0
            test()
            for i in range(int(mnist.train.num_examples / batch_size)):
                epoch_x, epoch_y = mnist.train.next_batch(batch_size)
                _, c = sess.run([optimizer, cost], feed_dict={x: epoch_x,
                                                              y: epoch_y})
                epoch_loss += c
                total = int(mnist.train.num_examples / batch_size)
                print("%d of %d" % (i, total))
                if epoch % 100 == 0:
                    test()
            print("Epoch %d, loss: %f" % (epoch, epoch_loss))
            test()


# if __name__ == "__main__":
#     test_discriminator()
