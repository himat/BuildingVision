import tensorflow as tf

# Tensorflow layer shortcuts

def conv(x, filters, stride=2, kernel_size=4):
    layer = tf.layers.conv2d(x, filters, kernel_size,
        strides=stride, padding='same',
        kernel_initializer=tf.random_normal_initializer(mean=0.0, stddev=0.02))
    return layer

def batchNorm(x, decay=0.99):
    layer = tf.contrib.layers.batch_norm(x, decay=decay,
        is_training=True, updates_collections=None)
    return layer

def leakyRelu(x, alpha=0.2):
    return tf.maximum(x * alpha, x)

def up_conv(x, filters, stride=2, kernel_size=4):
    layer = tf.layers.conv2d_transpose(x, filters, kernel_size,
        strides=stride, padding='same',
        kernel_initializer=tf.random_normal_initializer(mean=0.0, stddev=0.02))
    return layer

def dropout(x, rate=0.5):
    return tf.layers.dropout(x, rate, training=True)

def relu(x):
    return tf.nn.relu(x)


# Macroscopic U-net encoder-decoder layers

def encoder_layer(x, filters):
    x = conv(x, filters)
    return leakyRelu(x)

def encoder_layer_batchnorm(x, filters):
    x = conv(x, filters)
    x = batchNorm(x)
    return leakyRelu(x)

def decoder_layer_dropout(x, concat, filters):
    if concat != None:
        x = tf.concat([x, concat], 3)
    x = up_conv(x, filters)
    x = batchNorm(x)
    x = dropout(x)
    return relu(x)

def decoder_layer(x, concat, filters):
    x = tf.concat([x, concat], 3)
    x = up_conv(x, filters)
    x = batchNorm(x)
    return relu(x)

def output_layer(x, concat, channels=3):
    x = tf.concat([x, concat], 3)
    x = up_conv(x, channels)
    return tf.tanh(x)


# Create U-net encoder-decoder network
def u_net(x):
    x = tf.reshape(x, [-1, 128, 128, 1])

    e1 = encoder_layer(x, 64)
    e2 = encoder_layer_batchnorm(e1, 128)
    e3 = encoder_layer_batchnorm(e2, 256)
    e4 = encoder_layer_batchnorm(e3, 512)
    e5 = encoder_layer_batchnorm(e4, 512)
    e6 = encoder_layer_batchnorm(e5, 512)
    e7 = encoder_layer_batchnorm(e6, 512)

    d8 = decoder_layer_dropout(e7, None, 512)
    d9 = decoder_layer_dropout(d8, e6, 512)
    d10 = decoder_layer_dropout(d9, e5, 512)
    d11 = decoder_layer(d10, e4, 256)
    d12 = decoder_layer(d11, e3, 128)
    d13 = decoder_layer(d12, e2, 64)

    return output_layer(d13, e1)


# Tests

def test():
    shape = [4, 128, 128, 1]
    x = tf.Variable(tf.random_normal(shape))
    print(u_net(x).shape)

if __name__ == '__main__':
    test()
