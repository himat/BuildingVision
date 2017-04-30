import tensorflow as tf


# general helper functions
def flatten(l):
    for i in l:
        for j in i:
            yield j


# Variables
def filters(input, output, shape=(4, 4)):
    (s1, s2) = shape
    return tf.Variable(tf.random_normal([s1, s2, input, output], stddev=0.02))


def bias(shape):
    return tf.Variable(tf.random_normal([shape], stddev=0.02))


def denseW(input, output):
    return tf.Variable(tf.random_normal([input, output], stddev=0.02))


def batchnorm_vars(channels):
    scale = tf.Variable(tf.random_normal([channels], mean=1.0, stddev=0.02))
    offset = tf.Variable(tf.zeros([channels]))
    mean = tf.Variable(tf.zeros([channels]), trainable=False)
    var = tf.Variable(tf.ones([channels]), trainable=False)
    return (offset, scale, mean, var)


# layers
def lrelu(x, a=0.2):
    return (0.5 * (1 + a)) * x + (0.5 * (1 - a)) * tf.abs(x)


# http://r2rt.com/implementing-batch-normalization-in-tensorflow.html#making-predictions-with-the-model
def batchnorm(x, batch_norm_vars, is_training, decay=0.99):
    epsilon = 1e-8
    offset, scale, pop_mean, pop_var = batch_norm_vars

    def train():
        batch_mean, batch_var = tf.nn.moments(x, axes=[0, 1, 2],
                                              keep_dims=False)
        train_mean = tf.assign(pop_mean,
                               pop_mean * decay + batch_mean * (1 - decay))
        train_var = tf.assign(pop_var,
                              pop_var * decay + batch_var * (1 - decay))
        with tf.control_dependencies([train_mean, train_var]):
            return tf.nn.batch_normalization(x, batch_mean, batch_var,
                                             offset, scale, epsilon)

    def not_train():
        return tf.nn.batch_normalization(x, pop_mean, pop_var,
                                         offset, scale, epsilon)

    return tf.cond(is_training, train, not_train)


def conv(x, id, W, b, bv, is_training, strides=2, decay=0.99):
    x = tf.nn.conv2d(x, W,
                     strides=[1, strides, strides, 1], padding='SAME')
    x = tf.nn.bias_add(x, b)
    x = lrelu(x, a=0.2)
    x = batchnorm(x, bv, is_training, decay=decay)
    return x


def maxpool(x, k=2):
    return tf.nn.max_pool(x, ksize=[1, k, k, 1], strides=[1, k, k, 1],
                          padding='SAME')


def dense(x, W, b):
    x = tf.matmul(x, W)
    x = tf.add(x, b)
    return tf.nn.sigmoid(x)


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

    batch_vars = {
        'c1': batchnorm_vars(64),
        'c2': batchnorm_vars(128),
        'c3': batchnorm_vars(256),
        'c4': batchnorm_vars(512),
        'c5': batchnorm_vars(512),
        'c6': batchnorm_vars(512),
        'c7': batchnorm_vars(1),
    }

    return (weights, biases, batch_vars)


def conv_net(x, vars, is_training):
    weights, biases, batch_vars = vars
    x = tf.reshape(x, shape=[-1, 128, 128, 4])

    c1 = conv(x, 1, weights['c1'], biases['c1'], batch_vars['c1'],
              is_training, strides=2)

    c2 = conv(c1, 2, weights['c2'], biases['c2'], batch_vars['c2'],
              is_training, strides=2)

    c3 = conv(c2, 3, weights['c3'], biases['c3'], batch_vars['c3'],
              is_training, strides=2)

    c4 = conv(c3, 4, weights['c4'], biases['c4'], batch_vars['c4'],
              is_training, strides=2)

    c5 = conv(c4, 5, weights['c5'], biases['c5'], batch_vars['c5'],
              is_training, strides=2)

    c6 = conv(c5, 6, weights['c6'], biases['c6'], batch_vars['c6'],
              is_training, strides=2)

    c7 = conv(c6, 7, weights['c7'], biases['c7'], batch_vars['c7'],
              is_training, strides=2)

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

    batch_vars = {
        'c1': batchnorm_vars(32),
        'c2': batchnorm_vars(128),
        'c3': batchnorm_vars(256),
    }

    x = tf.placeholder(tf.float32, [None, 784], name='x')
    y = tf.placeholder(tf.float32, [None, n_classes], name='y')
    is_training = tf.placeholder(tf.bool, [], name='is_training')

    v = tf.reshape(x, shape=[-1, 28, 28, 1])

    c1 = conv(v, 1, weights['c1'],
              biases['c1'], batch_vars['c1'], is_training, strides=2)

    c2 = conv(c1, 2, weights['c2'],
              biases['c2'], batch_vars['c2'], is_training, strides=2)

    c3 = conv(c2, 3, weights['c3'],
              biases['c3'], batch_vars['c3'], is_training, strides=2)

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
                                              y: mnist.test.labels[:256],
                                              is_training: True}))

        for epoch in range(hm_epochs):
            epoch_loss = 0
            test()
            for i in range(int(mnist.train.num_examples / batch_size)):
                epoch_x, epoch_y = mnist.train.next_batch(batch_size)
                _, c = sess.run([optimizer, cost],
                                feed_dict={x: epoch_x, y: epoch_y,
                                           is_training: False})
                epoch_loss += c
                total = int(mnist.train.num_examples / batch_size)
                print("%d of %d" % (i, total))
                if epoch % 100 == 0:
                    test()
            print("Epoch %d, loss: %f" % (epoch, epoch_loss))
            test()


if __name__ == "__main__":
    test_discriminator()
