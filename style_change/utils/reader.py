import glob
import os
import tensorflow as tf


def image_decode(data_dir, height, width, format_, preprocess_fn, epochs):
    filenames = glob.glob(os.path.join(data_dir, '*.'+format_))
    filename_queue = tf.train.string_input_producer(filenames,
                                                    shuffle=True,
                                                    num_epochs=epochs)

    reader = tf.WholeFileReader()

    _, value = reader.read(filename_queue)

    if format_ == 'png':
        image_origin = tf.image.decode_png(value, channels=3)
    if format_ in ['jpg', 'jpeg']:
        image_origin = tf.image.decode_jpeg(value, channels=3)
    else:
        raise ValueError('Unknown image format...')

    return preprocess_fn(image_origin, height, width)


def image_input(data_dir, height, width, format_, batch_size, preprocess_fn, epochs):
    image = image_decode(data_dir, height, width, format_, preprocess_fn, epochs)

    return tf.train.batch([image],
                          batch_size,
                          dynamic_pad=True)
