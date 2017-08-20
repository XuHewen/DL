import tensorflow as tf
from utils import utils


def conv2d(x, input_filters, output_filters, kernel_size, strides, mode='REFLECT'):
    shape = [kernel_size, kernel_size, input_filters, output_filters]
    W = utils.variable_with_weight_decay('W',
                                         shape)
    x_padded = tf.pad(x,
                      [[0, 0], [kernel_size//2, kernel_size//2],
                       [kernel_size//2, kernel_size//2], [0, 0]],
                      mode=mode)
    conv = tf.nn.conv2d(x_padded, W, strides=[1, strides, strides, 1],
                        padding='VALID', name='conv')

    return conv


def conv2d_transpose(x, input_filters, output_filters, kernel_size, strides):
    shape = [kernel_size, kernel_size, output_filters, input_filters]
    W = utils.variable_with_weight_decay('W',
                                         shape)

    output_shape = tf.stack([tf.shape(x)[0],
                             tf.shape(x)[1] * strides,
                             tf.shape(x)[2] * strides,
                             output_filters])
    conv2d_transpose = tf.nn.conv2d_transpose(x,
                                              W,
                                              output_shape,
                                              strides=[1, strides, strides, 1],
                                              padding='SAME',
                                              name='conv_transpose')
    return conv2d_transpose


def instance_norm(x):
    epsilon = 1e-9
    mean, var = tf.nn.moments(x, [1, 2], keep_dims=True)
    return tf.div(tf.subtract(x, mean), tf.sqrt(tf.add(var, epsilon)))


def batch_norm(x, size, is_training, decay=0.999):
    beta = tf.Variable(tf.zeros([size]), name='beta')
    scale = tf.Variable(tf.zeros([size]), name='scale')
    pop_mean = tf.Variable(tf.zeros[size])
    pop_var = tf.Variable(tf.zeros(size))

    epsilon = 1e-3

    batch_mean, batch_var = tf.nn.moments(x, [0, 1, 2])
    train_mean = tf.assign(pop_mean, pop_mean * decay + batch_mean * (1 - decay))
    train_var = tf.assign(pop_var, pop_var * decay + batch_var * (1 - decay))

    def batch_statistics():
        with tf.control_dependencies([train_mean, train_var]):
            return tf.nn.batch_normalization(x,  batch_mean, batch_var,
                                             beta, scale, epsilon, name='batch_norm')

    def poplation_statistics():
        return tf.nn.batch_normalization(x, pop_mean, pop_var, beta, scale,
                                         epsilon, name='batch_norm')

    return tf.cond(is_training, batch_statistics, poplation_statistics)


def relu(input):
    relu = tf.nn.relu(input)
    # convert nan to zero (nan != nan)
    nan_to_zero = tf.where(tf.equal(relu, relu), relu, tf.zeros_like(relu))
    return nan_to_zero


def residual(x, filters, kernel_size, strides):

    with tf.variable_scope('residual') as scope:
        conv1 = conv2d(x, filters, filters, kernel_size, strides)
        scope.reuse_variables()
        conv2 = conv2d(relu(conv1), filters, filters, kernel_size, strides)

    return x + conv2


def net(image):

    # Less border effects when padding a little before passing through...
    image = tf.pad(image, [[0, 0], [10, 10], [10, 10], [0, 0]], mode='REFLECT')

    with tf.variable_scope('conv1'):
        conv1 = relu(instance_norm(conv2d(image, 3, 32, 9, 1)))

    with tf.variable_scope('conv2'):
        conv2 = relu(instance_norm(conv2d(conv1, 32, 64, 3, 2)))

    with tf.variable_scope('conv3'):
        conv3 = relu(instance_norm(conv2d(conv2, 64, 128, 3, 2)))

    with tf.variable_scope('res1'):
        res1 = residual(conv3, 128, 3, 1)

    with tf.variable_scope('res2'):
        res2 = residual(res1, 128, 3, 1)

    with tf.variable_scope('res3'):
        res3 = residual(res2, 128, 3, 1)

    with tf.variable_scope('res4'):
        res4 = residual(res3, 128, 3, 1)

    with tf.variable_scope('res5'):
        res5 = residual(res4, 128, 3, 1)

    with tf.variable_scope('deconv1'):
        deconv1 = relu(instance_norm(conv2d_transpose(res5, 128, 64, 3, 2)))

    with tf.variable_scope('deconv2'):
        deconv2 = relu(instance_norm(conv2d_transpose(deconv1, 64, 32, 3, 2)))

    with tf.variable_scope('deconv3'):
        deconv3 = tf.nn.tanh(instance_norm(conv2d(deconv2, 32, 3, 9, 1)))

    y = (deconv3 + 1) * 127.5

    height = tf.shape(y)[1]
    width = tf.shape(y)[2]

    y = tf.slice(y, [0, 10, 10, 0], tf.stack([-1, height - 20, width - 20, -1]))

    return y
