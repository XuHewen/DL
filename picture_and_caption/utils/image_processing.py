import tensorflow as tf


def distort_image(image, thread_id):

    # Randomly flop left right (horizontally)
    with tf.name_scope('flip_horizontal', values=[image]):
        image = tf.image.random_flip_left_right(image)

    # Randomly distort the colors based on thread id
    color_ordering = thread_id % 2
    with tf.name_scope('distort_color', values=[image]):
        if color_ordering == 0:
            image = tf.image.random_brightness(image, max_delta=32./255.)
            image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
            image = tf.image.random_hue(image, max_delta=0.032)
            image = tf.image.random_contrast(image, lower=0.5, upper=1.5)
        elif color_ordering == 1:
            image = tf.image.random_brightness(image, max_delta=32. / 255.)
            image = tf.image.random_contrast(image, lower=0.5, upper=1.5)
            image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
            image = tf.image.random_hue(image, max_delta=0.032)
        image = tf.clip_by_value(image, 0.0, 0.1)
    return image


def process_image(encoded_image, is_training, height, width,
                  resize_height=346, resize_width=346, thread_id=0, image_format='jpeg'):

    def image_summary(name, image):
        if thread_id == 0:
            tf.summary.image(name, tf.expand_dims(image, 0))

    with tf.name_scope('decode', values=[encoded_image]):
        if image_format == 'jpeg':
            image = tf.image.decode_jpeg(encoded_image, channels=3)
        elif image_format == 'png':
            image = tf.image.decode_png(encoded_image, channels=3)
        else:
            raise ValueError('Invalid image format: %s' % image_format)

    image = tf.image.convert_image_dtype(image, dtype=tf.float32)
    image_summary('original_image', image)

    image = tf.image.resize_images(image,
                                   size=[resize_height, resize_width],
                                   method=tf.image.ResizeMethod.BILINEAR)

    if is_training:
        image = tf.random_crop(image, [height, width, 3])
    else:
        image = tf.image.resize_image_with_crop_or_pad(image, height, width)

    image_summary('resize_image', image)

    if is_training:
        image = distort_image(image, thread_id)

    image_summary("final_image", image)

    image = tf.subtract(image, 0.5)
    image = tf.multiply(image, 2.0)

    return image


def main(_):
    filename = 'test.jpg'

    with tf.gfile.FastGFile(filename, 'rb') as f:
        image_byte = f.read()

        image = process_image(image_byte, True, 224, 224)
        # image = tf.image.decode_jpeg(image_byte, channels=3)
        #
        # image_1 = tf.image.adjust_brightness(image, delta=0.9)
        #
        # image_2 = tf.image.adjust_saturation(image, saturation_factor=5)
        #
        # image_3 = tf.image.adjust_hue(image, delta=0.9)
        #
        # image_4 = tf.image.adjust_contrast(image, contrast_factor=0.1)
        #
        # image_5 = tf.image.convert_image_dtype(image, dtype=tf.float32)
        # image_5 = tf.clip_by_value(image, 0.0, 1.0)

    with tf.Session() as sess:

        image_ = sess.run(image)

        import matplotlib.pyplot as plt
        plt.figure(1)
        plt.imshow(image_)
        plt.axis('off')

        plt.show()


if __name__ == '__main__':
    tf.app.run()
