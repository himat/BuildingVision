import tensorflow as tf


# Variable initialization

def filters(in_filters, out_filters, kernel_size=4):
    shape = [kernel_size, kernel_size, in_filters, out_filters]
    return tf.Variable(tf.random_normal(shape, stddev=0.02))

def bias(filters):
    return tf.Variable(tf.random_normal([filters], stddev=0.02))


# Tensorflow layer shortcuts

def conv(x, kernels, bias, stride=2):
    strides = [1, stride, stride, 1]
    x = tf.nn.conv2d(x, kernels, strides, 'SAME')
    return tf.nn.bias_add(x, bias)

def batchNorm(x, decay=0.99, is_training=True):
    return tf.contrib.layers.batch_norm(x, decay=decay,
        is_training=is_training, updates_collections=None)

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

def dropout(x, rate):
    # rate = tf.Print(rate, [rate])
    return tf.nn.dropout(x, rate)

def relu(x):
    return tf.nn.relu(x)


# U-net encoder-decoder modules

def encoder_layer(x, kernels, bias):
    x = conv(x, kernels, bias)
    return leakyRelu(x)

def encoder_layer_batchnorm(x, kernels, bias, is_training):
    x = conv(x, kernels, bias)
    x = batchNorm(x, is_training=is_training)
    return leakyRelu(x)

def decoder_layer_dropout(x, concat, kernels, bias, is_training, dropout_rate):
    if concat != None:
        x = tf.concat([x, concat], 3)
    x = up_conv(x, kernels, bias)
    x = batchNorm(x, is_training=is_training)
    x = dropout(x, rate=dropout_rate)
    return relu(x)

def decoder_layer(x, concat, kernels, bias, is_training):
    x = tf.concat([x, concat], 3)
    x = up_conv(x, kernels, bias)
    x = batchNorm(x, is_training=is_training)
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

        self.weights = self.w + self.b


    # Evaluate network given input
    def __call__(self, x, is_training, dropout_rate):
        x = tf.reshape(x, [-1, 128, 128, 1])
        w, b = self.w, self.b

        e1 = encoder_layer(x, w[0], b[0])
        e2 = encoder_layer_batchnorm(e1, w[1], b[1], is_training)
        e3 = encoder_layer_batchnorm(e2, w[2], b[2], is_training)
        e4 = encoder_layer_batchnorm(e3, w[3], b[3], is_training)
        e5 = encoder_layer_batchnorm(e4, w[4], b[4], is_training)
        e6 = encoder_layer_batchnorm(e5, w[5], b[5], is_training)
        e7 = encoder_layer_batchnorm(e6, w[6], b[6], is_training)

        d8 = decoder_layer_dropout(e7, None, w[7], b[7], is_training, dropout_rate)
        d9 = decoder_layer_dropout(d8, e6, w[8], b[8], is_training, dropout_rate)
        d10 = decoder_layer_dropout(d9, e5, w[9], b[9], is_training, dropout_rate)
        d11 = decoder_layer(d10, e4, w[10], b[10], is_training)
        d12 = decoder_layer(d11, e3, w[11], b[11], is_training)
        d13 = decoder_layer(d12, e2, w[12], b[12], is_training)

        return output_layer(d13, e1, w[13], b[13])



# Tests

def test():
    shape = [4, 128, 128, 1]
    x = tf.Variable(tf.random_normal(shape))
    generator = Generator()
    print(generator(x).shape)

if __name__ == '__main__':
    test()
