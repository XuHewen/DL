import tensorflow as tf


slim = tf.contrib.slim


def vgg_16(images,
           trainable=True,
           is_training=True,
           weight_decay=4e-5,
           stddev=0.1,
           dropout_keep_prob=0.8,
           use_batch_norm=True,
           batch_norm_params=None,
           add_summaries=True,
           spatial_squeeze=True,
           scope='vgg_16'):

    is_vgg16_model_training = trainable and is_training

    if use_batch_norm:
        # default parameters for batch normalization
        if not batch_norm_params:
            batch_norm_params = {
                'is_training': is_vgg16_model_training,
                'trainable': trainable,
                'decay': 0.9997,
                'epsilon': 0.001,
                'variables_collections': {
                    'beta': None,
                    'gamma': None,
                    'moving_mean': ['moving_vars'],
                    'moving_variance': ['moving_vars']
                }
            }

    if trainable:
        weights_regularizer = slim.l2_regularizer(weight_decay)
    else:
        weights_regularizer = None

    with tf.variable_scope(scope, 'vgg_16', [images]) as scope:
        end_points_collection = scope.name + '_end_points'
        with slim.arg_scope([slim.conv2d, slim.fully_connected],
                            weights_regularizer=weights_regularizer,
                            trainable=trainable,
                            outputs_collections=end_points_collection):
            with slim.arg_scope(
                [slim.conv2d],
                weights_initializer=tf.truncated_normal_initializer(stddev=stddev),
                activation_fn=tf.nn.relu
                # normalizer_fn=slim.batch_norm,
                # normalizer_params=batch_norm_params
            ):
                net = slim.repeat(images, 2, slim.conv2d, 64, [3, 3], scope='conv1')
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
                # Convert end_points_collection into a end_point dict.
                end_points = slim.utils.convert_collection_to_dict(
                    end_points_collection)
                if spatial_squeeze:
                    net = tf.squeeze(net, [1, 2], name='fc7/squeezed')
                    end_points[scope.name + '/fc7'] = net

    if add_summaries:
        for v in end_points.values():
            tf.contrib.layers.summarize_activation(v)

    return net


# test
def main(_):
    with tf.Graph().as_default():
        with tf.gfile.FastGFile('lena.png', 'rb') as f:
            image_bytes = f.read()

        image = tf.image.decode_png(image_bytes, channels=3)
        image = tf.image.convert_image_dtype(image, dtype=tf.float32)
        image = tf.image.resize_images(image,
                                       size=[224, 224],
                                       method=tf.image.ResizeMethod.BILINEAR)
        images = tf.expand_dims(image, 0)

        net, end_points = vgg_16(images)

        init_fn = slim.assign_from_checkpoint_fn(
            './pre_vgg_ckpt/vgg_16.ckpt',
            slim.get_model_variables('vgg_16'),
            ignore_missing_vars=True
        )

        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            init_fn(sess)
            end_points_dict = sess.run(end_points)

            print(end_points_dict['vgg_16/conv1/conv1_1'])


if __name__ == '__main__':
    tf.app.run()
