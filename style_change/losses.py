import tensorflow as tf
from nets import nets_factory
from preprocessing import preprocessing_factory
from utils import utils
import os

slim = tf.contrib.slim


def gram(layer):
    shape = tf.shape(layer)
    num_images = shape[0]
    width = shape[1]
    height = shape[2]
    num_filters = shape[3]
    filters = tf.reshape(layer, tf.stack([num_images, -1, num_filters]))
    x = tf.to_float(width * height * num_filters)
    grams = tf.matmul(filters, filters, transpose_a=True) / x
    return grams


def get_style_features(FLAGS):

    with tf.Graph().as_default():
        net_fn = nets_factory.get_network_fn(
            FLAGS.loss_model, num_classes=1, is_training=False)

        pre_fn, pro_fn = preprocessing_factory.get_preprocessing(
            FLAGS.loss_model, is_training=False)

        size = FLAGS.image_size
        img_bytes = tf.read_file(FLAGS.style_image)

        if FLAGS.style_image.lower().endswith('png'):
            image = tf.image.decode_png(img_bytes)
        else:
            image = tf.image.decode_jpeg(img_bytes)

        image = pre_fn(image, size, size)
        images = tf.expand_dims(image, 0)

        _, endpoints_dict = net_fn(images, spatial_squeeze=False)

        features = []
        for layer in FLAGS.style_layers:
            feature = endpoints_dict[layer]
            feature = tf.squeeze(gram(feature), [0])
            features.append(feature)

        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())

            init_fn = utils._get_init_fn(FLAGS)
            init_fn(sess)

            # Make sure the 'generated' directory exists
            if not os.path.exists('generated'):
                os.makedirs('generated')
            save_file = 'generated/target_style_' + FLAGS.naming + '.jpg'

            with open(save_file, 'wb') as f:
                target_image = pro_fn(images[0, :])
                value = tf.image.encode_jpeg(tf.cast(target_image, tf.uint8))
                f.write(sess.run(value))
                tf.logging.info(
                    'Target style pattern is saved to: %s.' % save_file)

            return sess.run(features)


def style_loss(endpoints_dict, style_features_t, style_layers):
    style_loss = 0
    style_loss_summary = {}

    for style_gram, layer in zip(style_features_t, style_layers):
        generated_images, _ = tf.split(endpoints_dict[layer], 2)
        size = tf.size(generated_images)
        layer_style_loss = tf.nn.l2_loss(
            gram(generated_images) - style_gram) / tf.to_float(size)
        style_loss_summary[layer] = layer_style_loss
        style_loss += layer_style_loss

    return style_loss, style_loss_summary


def content_loss(endpoints_dict, content_layers):
    content_loss = 0
    for layer in content_layers:
        generated_images, content_images = tf.split(endpoints_dict[layer], 2, 0)
        size = tf.size(generated_images)
        content_loss = tf.nn.l2_loss(
            generated_images - content_images) / tf.to_float(size)
    return content_loss


def total_variation_loss(layer):
    shape = tf.shape(layer)
    height = shape[1]
    width = shape[2]

    y = tf.slice(layer, [0, 0, 0, 0], [-1, height - 1, -1, -1]) - tf.slice(
        layer, [0, 1, 0, 0], [-1, -1, -1, -1])
    x = tf.slice(layer, [0, 0, 0, 0], [-1, -1, width - 1, -1]) - tf.slice(
        layer, [0, 0, 1, 0], [-1, -1, -1, -1])
    loss = tf.nn.l2_loss(x) / tf.to_float(
        tf.size(x)) + tf.nn.l2_loss(y) / tf.to_float(tf.size(y))
    return loss
