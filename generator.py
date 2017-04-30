import tensorflow as tf


# Variable initialization

def filters(in_filters, out_filters, kernel_size=4):
    shape = [kernel_size, kernel_size, in_filters, out_filters]
    return tf.Variable(tf.random_normal(shape, stddev=0.02))

def bias(filters):
    return tf.Variable(tf.random_normal([filters], stddev=0.02))

def pop_vars(channels):
    mean = tf.Variable(tf.zeros([channels]), trainable=False)
    var = tf.Variable(tf.ones([channels]), trainable=False)
    return (mean, var)

def batchnorm_vars(channels):
    offset = tf.Variable(tf.zeros([channels]))
    scale = tf.Variable(tf.random_normal([channels], mean=1.0, stddev=0.02))
    return (offset, scale)


# Tensorflow layer shortcuts

def conv(x, kernels, bias, stride=2):
    strides = [1, stride, stride, 1]
    x = tf.nn.conv2d(x, kernels, strides, 'SAME')
    return tf.nn.bias_add(x, bias)

# http://r2rt.com/implementing-batch-normalization-in-tensorflow.html#making-predictions-with-the-model
def batchnorm(x, pop_vars, batch_norm_vars, is_training, decay=0.99):
    epsilon = 1e-8
    offset, scale = batch_norm_vars
    pop_mean, pop_var = pop_vars

    if is_training:
        batch_mean, batch_var = tf.nn.moments(x, axes=[0, 1, 2], keep_dims=False)
        train_mean = tf.assign(pop_mean,
                               pop_mean * decay + batch_mean * (1 - decay))
        train_var = tf.assign(pop_var,
                              pop_var * decay + batch_var * (1 - decay))
        with tf.control_dependencies([train_mean, train_var]):
            return tf.nn.batch_normalization(x, batch_mean, batch_var,
                offset, scale, epsilon)
    else:
        return tf.nn.batch_normalization(x, pop_mean, pop_var,
            offset, scale, epsilon)

def leakyRelu(x, alpha=0.2):
    return tf.maximum(x * alpha, x)

def up_conv(x, kernels, bias, stride=2):
    in_shape = x.shape.as_list()
    batch = in_shape[0]
    size = in_shape[1]*2
    channels = kernels.get_shape().as_list()[2]
    out_shape = [batch, size, size, channels]

    strides = [1, stride, stride, 1]
    x = tf.nn.conv2d_transpose(x, kernels, out_shape, strides, 'SAME')
    return tf.nn.bias_add(x, bias)

def dropout(x, rate=0.5):
    return tf.nn.dropout(x, rate)

def relu(x):
    return tf.nn.relu(x)


# U-net encoder-decoder modules

def encoder_layer(x, kernels, bias):
    x = conv(x, kernels, bias)
    return leakyRelu(x)

def encoder_layer_batchnorm(x, kernels, bias, pop_vars, batch_norm_vars, is_training):
    x = conv(x, kernels, bias)
    x = batchnorm(x, pop_vars, batch_norm_vars, is_training)
    return leakyRelu(x)

def decoder_layer_dropout(x, concat, kernels, bias, pop_vars, batch_norm_vars, is_training):
    if concat != None:
        x = tf.concat([x, concat], 3)
    x = up_conv(x, kernels, bias)
    x = batchnorm(x, pop_vars, batch_norm_vars, is_training)
    if is_training:
        x = dropout(x)
    return relu(x)

def decoder_layer(x, concat, kernels, bias, pop_vars, batch_norm_vars, is_training):
    x = tf.concat([x, concat], 3)
    x = up_conv(x, kernels, bias)
    x = batchnorm(x, pop_vars, batch_norm_vars, is_training)
    return relu(x)

def output_layer(x, concat, kernels, bias):
    x = tf.concat([x, concat], 3)
    x = up_conv(x, kernels, bias)
    return tf.tanh(x)



class Generator(object):

    # Create generator network (U-net encoder-decoder)
    def __init__(self):
        self.w = [
            # Encoder
            filters(1, 64),
            filters(64, 128),
            filters(128, 256),
            filters(256, 512),
            filters(512, 512),
            filters(512, 512),
            filters(512, 512),

            # Decoder
            filters(512, 512),
            filters(512, 1024),
            filters(512, 1024),
            filters(256, 1024),
            filters(128, 512),
            filters(64, 256),
            filters(3, 128)
        ]

        self.b = [
            # Encoder
            bias(64),
            bias(128),
            bias(256),
            bias(512),
            bias(512),
            bias(512),
            bias(512),

            # Decoder
            bias(512),
            bias(512),
            bias(512),
            bias(256),
            bias(128),
            bias(64),
            bias(3)
        ]

        self.p = [
            pop_vars(128),
            pop_vars(256),
            pop_vars(512),
            pop_vars(512),
            pop_vars(512),
            pop_vars(512),

            pop_vars(512),
            pop_vars(512),
            pop_vars(512),
            pop_vars(256),
            pop_vars(128),
            pop_vars(64),
        ]

        self.n = [
            batchnorm_vars(128),
            batchnorm_vars(256),
            batchnorm_vars(512),
            batchnorm_vars(512),
            batchnorm_vars(512),
            batchnorm_vars(512),

            batchnorm_vars(512),
            batchnorm_vars(512),
            batchnorm_vars(512),
            batchnorm_vars(256),
            batchnorm_vars(128),
            batchnorm_vars(64),
        ]

        self.weights = (self.w + self.b
                        + [x[1] for x in self.p]
                        + [x[0] for x in self.p]
                        + [x[1] for x in self.n]
                        + [x[0] for x in self.n])


    # Evaluate network given input
    def __call__(self, x, is_training=True):
        x = tf.reshape(x, [-1, 128, 128, 1])
        w, b, p, n = self.w, self.b, self.p, self.n

        e1 = encoder_layer(x, w[0], b[0])
        e2 = encoder_layer_batchnorm(e1, w[1], b[1], p[0], n[0], is_training)
        e3 = encoder_layer_batchnorm(e2, w[2], b[2], p[1], n[1], is_training)
        e4 = encoder_layer_batchnorm(e3, w[3], b[3], p[2], n[2], is_training)
        e5 = encoder_layer_batchnorm(e4, w[4], b[4], p[3], n[3], is_training)
        e6 = encoder_layer_batchnorm(e5, w[5], b[5], p[4], n[4], is_training)
        e7 = encoder_layer_batchnorm(e6, w[6], b[6], p[5], n[5], is_training)

        d8 = decoder_layer_dropout(e7, None, w[7], b[7], p[6], n[6], is_training)
        d9 = decoder_layer_dropout(d8, e6, w[8], b[8], p[7], n[7], is_training)
        d10 = decoder_layer_dropout(d9, e5, w[9], b[9], p[8], n[8], is_training)
        d11 = decoder_layer(d10, e4, w[10], b[10], p[9], n[9], is_training)
        d12 = decoder_layer(d11, e3, w[11], b[11], p[10], n[10], is_training)
        d13 = decoder_layer(d12, e2, w[12], b[12], p[11], n[11], is_training)

        return output_layer(d13, e1, w[13], b[13])



# Tests

def test():
    shape = [4, 128, 128, 1]
    x = tf.Variable(tf.random_normal(shape))
    generator = Generator()
    print(generator(x).shape)
    print(generator(x, is_training=False).shape)

if __name__ == '__main__':
    test()
