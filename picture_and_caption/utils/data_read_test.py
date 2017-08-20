import tensorflow as tf
import glob


class MyTFRecordReader:

    def __init__(self, filename=None, pattern=None):
        self.filename = filename
        self.pattern = pattern

    def read_tf(self, filequeue):
        reader = tf.TFRecordReader()
        _, serialized_example = reader.read(filequeue)

        context_features = {
            "image/image_id": tf.FixedLenFeature([], tf.int64),
            'image/data': tf.FixedLenFeature([], tf.string)
        }

        sequence_features = {
            "image/caption": tf.FixedLenSequenceFeature([], tf.string),
            "image/caption_ids": tf.FixedLenSequenceFeature([], tf.int64)
        }

        context_parsed, sequence_parsed = tf.parse_single_sequence_example(
            serialized=serialized_example,
            context_features=context_features,
            sequence_features=sequence_features
        )

        return context_parsed, sequence_parsed

    def from_record(self):
        if self.filename:
            filequeue = tf.train.string_input_producer([self.filename])
        elif self.pattern:
            filenames = glob.glob(self.pattern)
            filequeue = tf.train.string_input_producer(filenames)

        context, sequence = self.read_tf(filequeue)

        return context, sequence


with tf.Graph().as_default():
    reader = MyTFRecordReader(pattern='/home/xu/store/deep/im2txt/tfrecord/train*')
    context, sequence = reader.from_record()
    image = tf.image.decode_jpeg(context['image/data'], channels=3)
    image = tf.image.random_flip_up_down(image)
    caption = sequence['image/caption']
    caption_ids = sequence['image/caption_ids']

    with tf.Session() as sess:

        sess.run(tf.global_variables_initializer())

        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)

        image_, caption_, caption_ids_ = sess.run([image, caption, caption_ids])

        import matplotlib.pyplot as plt
        plt.imshow(image_)
        plt.axis('off')
        plt.title(caption_)
        print(caption_ids_)

        coord.request_stop()
        coord.join(threads)

plt.show()
