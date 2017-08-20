import matplotlib.pyplot as plt
import glob

import tensorflow as tf
# from preprocessing import vgg_preprocessing

# IMAGE_HEIGHT = 64
# IMAGE_WIDTH = 64
# # for vgg16
IMAGE_HEIGHT = 224
IMAGE_WIDTH = 224

NUM_CLASSES = 2
NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN = 17500
NUM_EXAMPLES_PER_EPOCH_FOR_EVAL = 7500


def process_image(image, is_training, height, width,
                  resize_height=250, resize_width=250):
    tf.summary.image('original_image', tf.expand_dims(image, 0))
    image = tf.image.convert_image_dtype(image, dtype=tf.float32)

    image = tf.image.resize_images(image,
                                   size=[resize_height, resize_width],
                                   method=tf.image.ResizeMethod.BILINEAR)
    if is_training:
        image = tf.random_crop(image, [height, width, 3])
    else:
        image = tf.image.resize_image_with_crop_or_pad(image, height, width)
    tf.summary.image('resize_image', tf.expand_dims(image, 0))
    #
    if is_training:
        image = distort_image(image)

    image = tf.image.per_image_standardization(image)

    tf.summary.image('final_image', tf.expand_dims(image, 0))

    return image


def distort_image(image):

    # Randomly flop left right (horizontally)
    with tf.name_scope('flip_horizontal', values=[image]):
        image = tf.image.random_flip_left_right(image)

    with tf.name_scope('distort_color', values=[image]):

        image = tf.image.random_brightness(image, max_delta=32./255.)
        image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
        image = tf.image.random_hue(image, max_delta=0.032)
        image = tf.image.random_contrast(image, lower=0.5, upper=1.5)

    return image


def read_and_decode(filename_queue, batch_size, is_training=True):
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)

    features = tf.parse_single_example(
        serialized_example,
        features={
            'image/encoded': tf.FixedLenFeature([], tf.string),
            'image/class/label': tf.FixedLenFeature([], tf.int64)
        })

    image = tf.image.decode_jpeg(features['image/encoded'], channels=3)
    label = tf.cast(features['image/class/label'], tf.int32)

    height = IMAGE_HEIGHT
    width = IMAGE_WIDTH
    image = process_image(image, is_training, height, width)

    num_preprocess_threads = 4
    images, labels = tf.train.batch(
        [image, label],
        batch_size=batch_size,
        num_threads=num_preprocess_threads,
        capacity=15)

    return images, tf.reshape(labels, [batch_size])


def main(_):
    # This code does not work, I don't know why
    # data_path = tf.train.match_filenames_once("./tfrecord/*.tfrecord")
    # filename_queue = tf.train.string_input_producer(data_path)

    filenames = glob.glob("./tfrecord/*.tfrecord")
    filename_queue = tf.train.string_input_producer(filenames)

    image = read_and_decode(filename_queue, 25)

    labels_to_names = {0: 'cat', 1: 'dog'}

    with tf.Session() as sess:

        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)

        img, label = sess.run(image)
        print(img.shape, label.shape)

        plt.figure()
        for i in range(25):
            plt.subplot(5, 5, i + 1)
            plt.imshow(img[i, :, :, :])
            plt.axis('off')
            plt.title(labels_to_names[label[i]])
        # plt.imshow(img, aspect='auto')
        plt.show()

    coord.request_stop()
    coord.join(threads)


if __name__ == '__main__':
    tf.app.run()
