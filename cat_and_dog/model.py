import glob
import tensorflow as tf
import read_tfrecord

flags = tf.app.flags
FLAGS = flags.FLAGS

# Basic model parameters
flags.DEFINE_integer('batch_size', 128,
                     """Number of images to process in a batch.""")
flags.DEFINE_string('data_dir', './input',
                    """Path to the dog_cat data directory""")
flags.DEFINE_boolean('use_fp16', False,
                     """Train the model using fp16.""")

NUM_CLASSES = read_tfrecord.NUM_CLASSES
NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN = read_tfrecord.NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN
NUM_EXAMPLES_PER_EPOCH_FOR_EVAL = read_tfrecord.NUM_EXAMPLES_PER_EPOCH_FOR_EVAL

NUM_EPOCHS_PER_DECAY = 40.0
INITIAL_LEARNING_RATE = 0.1
LEARNING_RATE_DECAY_FACTOR = 0.1
MOVING_AVERAGE_DECAY = 0.9999


def _variable_on_cpu(name, shape, initializer):
    dtype = tf.float16 if FLAGS.use_fp16 else tf.float32
    var = tf.get_variable(
        name, shape, initializer=initializer, dtype=dtype
    )
    return var


def _variable_with_weight_decay(name, shape, stddev, wd):
    dtype = tf.float16 if FLAGS.use_fp16 else tf.float32
    var = _variable_on_cpu(
        name,
        shape,
        tf.truncated_normal_initializer(
            stddev=stddev, dtype=dtype
        )
    )
    if wd is not None:
        weight_decay = tf.multiply(tf.nn.l2_loss(var), wd, name='weight_loss')
        tf.add_to_collection('losses', weight_decay)
    return var


def distored_inputs():

    if not FLAGS.data_dir:
        raise ValueError('Please supply a data_dir')

    filenames = glob.glob("/input/*train*.tfrecord")
    filename_queue = tf.train.string_input_producer(filenames)

    images, labels = read_tfrecord.read_and_decode(filename_queue,
                                                   FLAGS.batch_size)
    tf.summary.image('input', images)
    return images, labels


def eval_inputs():
    if not FLAGS.data_dir:
        raise ValueError('Please supply a data_dir')

    filenames = glob.glob("/home/xu/deep/cat_dog/tfrecord/*valid*.tfrecord")
    filename_queue = tf.train.string_input_producer(filenames)

    images, labels = read_tfrecord.read_and_decode(filename_queue,
                                                   FLAGS.batch_size, is_training=False)

    return images, labels


def inference(images, is_training=True):
    """

    """
    # conv1
    with tf.variable_scope('conv1') as scope:
        kernel = _variable_with_weight_decay(
            'weights',
            shape=[5, 5, 3, 64],
            stddev=5e-2,
            wd=0.0
        )
        conv = tf.nn.conv2d(images, kernel, [1, 1, 1, 1], padding='SAME')
        biases = _variable_on_cpu('biases', [64], tf.constant_initializer(0.0))
        bias = tf.nn.bias_add(conv, biases)
        # conv1 = tf.nn.relu(bias, name=scope.name)
        conv1 = tf.contrib.layers.batch_norm(bias,
                                             center=True,
                                             scale=True,
                                             is_training=is_training,
                                             scope=scope,
                                             activation_fn=tf.nn.relu)

    # pool1
    pool1 = tf.nn.max_pool(
        conv1,
        ksize=[1, 3, 3, 1],
        strides=[1, 2, 2, 1],
        padding='SAME',
        name='pool1'
    )

    # norm1
    # norm1 = tf.nn.lrn(
    #     pool1,
    #     4,
    #     bias=1.0,
    #     alpha=0.001 / 9.0,
    #     beta=0.75,
    #     name='norm1'
    # )

    # conv2
    with tf.variable_scope('conv2') as scope:
        kernel = _variable_with_weight_decay(
            'weights',
            shape=[5, 5, 64, 64],
            stddev=5e-2,
            wd=0.0
        )
        conv = tf.nn.conv2d(pool1, kernel, [1, 1, 1, 1], padding='SAME')
        biases = _variable_on_cpu('biases', [64], tf.constant_initializer(0.0))
        bias = tf.nn.bias_add(conv, biases)
        # conv2 = tf.nn.relu(bias, name=scope.name)
        conv2 = tf.contrib.layers.batch_norm(bias,
                                             center=True,
                                             scale=True,
                                             is_training=is_training,
                                             scope=scope,
                                             activation_fn=tf.nn.relu)

    # norm2
    # norm2 = tf.nn.lrn(
    #     conv2,
    #     4,
    #     bias=1.0,
    #     alpha=0.001 / 0.9,
    #     beta=0.75,
    #     name='norm2'
    # )

    # pool2
    pool2 = tf.nn.max_pool(
        conv2,
        ksize=[1, 3, 3, 1],
        strides=[1, 2, 2, 1],
        padding='SAME',
        name='pool2'
    )

    # conv3
    with tf.variable_scope('conv3') as scope:
        kernel = _variable_with_weight_decay(
            'weights',
            shape=[5, 5, 64, 64],
            stddev=5e-2,
            wd=0.0
        )
        conv = tf.nn.conv2d(pool2, kernel, [1, 1, 1, 1], padding='SAME')
        biases = _variable_on_cpu('biases', [64], tf.constant_initializer(0.0))
        bias = tf.nn.bias_add(conv, biases)
        # conv3 = tf.nn.relu(bias, name=scope.name)
        conv3 = tf.contrib.layers.batch_norm(bias,
                                             center=True,
                                             scale=True,
                                             is_training=is_training,
                                             scope=scope,
                                             activation_fn=tf.nn.relu)

    # norm3
    # norm3 = tf.nn.lrn(
    #     conv3,
    #     4,
    #     bias=1.0,
    #     alpha=0.001 / 9.0,
    #     beta=0.75,
    #     name='norm3'
    # )

    # pool3
    pool3 = tf.nn.max_pool(
        conv3,
        ksize=[1, 3, 3, 1],
        strides=[1, 2, 2, 1],
        padding='SAME',
        name='pool3'
    )

    # local3
    with tf.variable_scope('local3') as scope:
        reshape = tf.reshape(pool3, [FLAGS.batch_size, -1])
        dim = reshape.get_shape()[1].value
        weights = _variable_with_weight_decay(
            'weights',
            shape=[dim, 384],
            stddev=0.04,
            wd=0.004
        )
        biases = _variable_on_cpu('biases', [384],
                                  tf.constant_initializer(0.0))

        local3 = tf.matmul(reshape, weights) + biases

        local3 = tf.contrib.layers.batch_norm(local3,
                                              center=True,
                                              scale=True,
                                              is_training=is_training,
                                              scope=scope,
                                              activation_fn=tf.nn.relu)
        if is_training:
            local3 = tf.nn.dropout(local3,
                                   keep_prob=0.5,
                                   name='dropout1')

    # local4
    with tf.variable_scope('local4') as scope:
        weights = _variable_with_weight_decay(
            'weights',
            shape=[384, 192],
            stddev=0.04,
            wd=0.004
        )
        biases = _variable_on_cpu('biases', [192],
                                  tf.constant_initializer(0.0))
        local4 = tf.matmul(local3, weights) + biases

        local4 = tf.contrib.layers.batch_norm(local4,
                                              center=True,
                                              scale=True,
                                              is_training=is_training,
                                              scope=scope,
                                              activation_fn=tf.nn.relu)

        if is_training:
            local4 = tf.nn.dropout(local4,
                                   keep_prob=0.5,
                                   name='dropout2')

    # softmax
    with tf.variable_scope('softmax_linear') as scope:
        weights = _variable_with_weight_decay(
            'weights',
            shape=[192, NUM_CLASSES],
            stddev=1 / 192.0,
            wd=0.0
        )
        biases = _variable_on_cpu('biases', [NUM_CLASSES],
                                  tf.constant_initializer(0.0))
        softmax_linear = tf.add(tf.matmul(local4, weights),
                                biases,
                                name=scope.name)
        # softmax_linear = tf.contrib.layers.batch_norm(softmax_linear,
        #                                               center=True,
        #                                               scale=True,
        #                                               is_training=True,
        #                                               scope=scope,
        #                                               activation_fn=None)

    return softmax_linear


slim = tf.contrib.slim


def vgg_arg_scope(weight_decay=0.0005):
    with slim.arg_scope(
        [slim.conv2d, slim.fully_connected],
            activation_fn=tf.nn.relu,
            weights_regularizer=slim.l2_regularizer(weight_decay),
            biases_initializer=tf.zeros_initializer()):
        with slim.arg_scope([slim.conv2d], padding='SAME') as arg_sc:
            return arg_sc


def vgg_16(inputs,
           num_classes=2,
           stddev=0.001,
           is_training=True,
           dropout_keep_prob=0.5,
           spatial_squeeze=True,
           scope='vgg_16'):

    batch_norm_params = {
        'is_training': is_training,
        'trainable': True,
        'decay': 0.9997,
        'epsilon': 0.001,
        'variables_collections': {
            'beta': None,
            'gamma': None,
            'moving_mean': ['moving_vars'],
            'moving_variance': ['moving_vars']
        }
    }

    with tf.variable_scope(scope, 'vgg_16', [inputs]) as sc:
        end_points_collection = sc.name + '_end_points'

        with slim.arg_scope(
            [slim.conv2d, slim.fully_connected, slim.max_pool2d],
                outputs_collections=end_points_collection):
            # with slim.arg_scope(
            #     [slim.conv2d],
            #     weights_initializer=tf.truncated_normal_initializer(stddev=stddev),
            #     activation_fn=tf.nn.relu,
            #     normalizer_fn=slim.batch_norm,
            #     normalizer_params=batch_norm_params
            # ):
            net = slim.repeat(
                inputs, 2, slim.conv2d, 64, [3, 3], scope='conv1')
            net = slim.max_pool2d(net, [2, 2], scope='pool1')
            net = slim.repeat(net, 2, slim.conv2d, 128, [3, 3], scope='conv2')
            net = slim.max_pool2d(net, [2, 2], scope='pool2')
            net = slim.repeat(net, 3, slim.conv2d, 256, [3, 3], scope='conv3')
            net = slim.max_pool2d(net, [2, 2], scope='pool3')
            net = slim.repeat(net, 3, slim.conv2d, 512, [3, 3], scope='conv4')
            net = slim.max_pool2d(net, [2, 2], scope='pool4')
            net = slim.repeat(net, 3, slim.conv2d, 512, [3, 3], scope='conv5')
            net = slim.max_pool2d(net, [2, 2], scope='pool5')

            # Use conv2d instead of fully_connected layers.
            net = slim.conv2d(net, 4096, [7, 7], padding='VALID', scope='fc6')
            net = slim.dropout(
                net,
                dropout_keep_prob,
                is_training=is_training,
                scope='dropout6')
            net = slim.conv2d(net, 4096, [1, 1], scope='fc7')
            net = slim.dropout(
                net,
                dropout_keep_prob,
                is_training=is_training,
                scope='dropout7')

            net = slim.conv2d(
                net,
                num_classes, [1, 1],
                activation_fn=None,
                normalizer_fn=None,
                scope='fc8')

            end_points = slim.utils.convert_collection_to_dict(
                end_points_collection)
            if spatial_squeeze:
                net = tf.squeeze(net, [1, 2], name='fc8/squeezed')
                end_points[sc.name + '/fc8'] = net

            return net, end_points


def loss(logits, labels):
    labels = tf.cast(labels, tf.int64)

    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
        logits=logits, labels=labels, name='cross_entropy_per_example'
    )
    cross_entropy_mean = tf.reduce_mean(cross_entropy, name='cross_entropy')

    tf.add_to_collection('losses', cross_entropy_mean)
    total_loss = tf.add_n(tf.get_collection('losses'), name='total_loss')
    tf.summary.scalar('total_loss', total_loss)
    return total_loss


tf.logging.set_verbosity(tf.logging.INFO)


def main(argv=None):  # pylint: disable=unused-argument
    with tf.Graph().as_default():

        def _lr_fn(learning_rate, global_step):
            num_batches_per_epoch = NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN / FLAGS.batch_size
            decay_steps = int(num_batches_per_epoch * NUM_EPOCHS_PER_DECAY)

            return tf.train.exponential_decay(
                learning_rate,
                global_step,
                decay_steps,
                LEARNING_RATE_DECAY_FACTOR,
                staircase=True
            )
        global_step = tf.Variable(initial_value=0, trainable=False, name='global_step')

        images, labels = distored_inputs()
        # using vgg_16
        # with slim.arg_scope(vgg_arg_scope()):
        #     logits, _ = vgg_16(images)
        logits = inference(images)
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        print(update_ops)
        total_loss = loss(logits, labels)

        labels = tf.cast(labels, tf.int64)
        correct_predictions = tf.equal(tf.argmax(logits, 1), labels)
        accuracy = tf.reduce_mean(tf.cast(correct_predictions, tf.float32))
        tf.summary.scalar('accuracy', accuracy)

        train_op = tf.contrib.layers.optimize_loss(
            loss=total_loss,
            global_step=global_step,
            learning_rate=0.01,
            optimizer='Adam',
            learning_rate_decay_fn=_lr_fn,
            update_ops=update_ops
        )

        saver = tf.train.Saver()
        tf.contrib.slim.learning.train(
            train_op,
            '/output/cat_vs_dog',
            log_every_n_steps=10,
            global_step=global_step,
            number_of_steps=30000,
            saver=saver,
            save_interval_secs=100,
            save_summaries_secs=100
        )


if __name__ == '__main__':
    tf.app.run()
