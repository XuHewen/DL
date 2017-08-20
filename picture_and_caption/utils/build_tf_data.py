import tensorflow as tf
import numpy as np
import collections
import nltk
import json

import os
import sys
import random
import threading

from datetime import datetime

flags = tf.flags
FLAGS = flags.FLAGS

tf.flags.DEFINE_string("train_image_dir", "/home/xu/store/deep/im2txt/train2014",
                       "Training image directory.")
tf.flags.DEFINE_string("val_image_dir", "/home/xu/store/deep/im2txt/val2014",
                       "Validation image directory.")

tf.flags.DEFINE_string("train_captions_file",
                       "/home/xu/store/deep/im2txt/annotations/captions_train2014.json",
                       "Training captions JSON file.")
tf.flags.DEFINE_string("val_captions_file",
                       "/home/xu/store/deep/im2txt/annotations/captions_val2014.json",
                       "Validation captions JSON file.")

tf.flags.DEFINE_string("output_dir", "/home/xu/store/deep/im2txt/tfrecord",
                       "Output data directory.")

tf.flags.DEFINE_integer("train_shards", 256,
                        "Number of shards in training TFRecord files.")
tf.flags.DEFINE_integer("val_shards", 4,
                        "Number of shards in validation TFRecord files.")
tf.flags.DEFINE_integer("test_shards", 8,
                        "Number of shards in testing TFRecord files.")

tf.flags.DEFINE_string("start_word", "<S>",
                       "Special word added to the beginning of each sentence.")
tf.flags.DEFINE_string("end_word", "</S>",
                       "Special word added to the end of each sentence.")
tf.flags.DEFINE_string("unknown_word", "<UNK>",
                       "Special word meaning 'unknown'.")
tf.flags.DEFINE_integer(
    "min_word_count", 4,
    "The minimum number of occurrences of each word in the "
    "training set for inclusion in the vocabulary.")
tf.flags.DEFINE_string("word_counts_output_file",
                       "/home/xu/store/deep/im2txt/annotations/word_counts.txt",
                       "Output vocabulary file of word counts.")

tf.flags.DEFINE_integer("num_threads", 4,
                        "Number of threads to preprocess the images.")


ImageMetaData = collections.namedtuple("ImageMetaData",
                                       ["image_id", "filename", "captions"])


class ImageDecoder(object):

    def __init__(self):

        self._sess = tf.Session()
        self._encoded_jpeg = tf.placeholder(tf.string)
        self._decode_jpeg = tf.image.decode_jpeg(self._encoded_jpeg,
                                                 channels=3)

    def decode_jpeg(self, encoded_jpeg):
        image = self._sess.run(self._decode_jpeg,
                               feed_dict={self._encoded_jpeg: encoded_jpeg})

        assert len(image.shape) == 3
        assert image.shape[2] == 3

        return image


class Vocabulary(object):

    def __init__(self, vocab, unk_id):
        self._vocab = vocab
        self._unk_id = unk_id

    def word_to_id(self, word):
        if word in self._vocab:
            return self._vocab[word]
        else:
            return self._unk_id


def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def _int64_feature_list(value):
    return tf.train.FeatureList(feature=[_int64_feature(v) for v in value])


def _bytes_feature_list(value):
    return tf.train.FeatureList(feature=[_bytes_feature(v) for v in value])


def _to_sequence_example(image, decoder, vocab):
    with tf.gfile.FastGFile(image.filename, 'rb') as f:
        encoded_image = f.read()
    try:
        decoder.decode_jpeg(encoded_image)
    except(tf.errors.InvalidArgumentError, AssertionError):
        print('skipping file with invalid JPEG data: %s' % image.filename)
        return

    context = tf.train.Features(feature={
        'image/image_id': _int64_feature(image.image_id),
        'image/data': _bytes_feature(encoded_image)
    })

    assert len(image.captions) == 1
    caption = image.captions[0]
    caption_ids = [vocab.word_to_id(word) for word in caption]
    feature_lists = tf.train.FeatureLists(feature_list={
        'image/caption': _bytes_feature_list(caption),
        'image/caption_ids': _int64_feature_list(caption_ids)
    })

    sequence_example = tf.train.SequenceExample(context=context,
                                                feature_lists=feature_lists)

    return sequence_example


def _process_image_files(thread_index, ranges, name, images, decoder,
                         vocab, num_shards):
    num_threads = len(ranges)
    assert not num_shards % num_threads

    num_shards_per_batch = int(num_shards / num_threads)

    shard_ranges = np.linspace(ranges[thread_index][0],
                               ranges[thread_index][1],
                               num_shards_per_batch + 1).astype(np.int)

    num_images_in_thread = ranges[thread_index][1] - ranges[thread_index][0]

    counter = 0

    for s in range(num_shards_per_batch):
        shard = thread_index * num_shards_per_batch + s

        output_filename = '%s-%.5d-of-%.5d' % (name, shard, num_shards)
        output_file = os.path.join(FLAGS.output_dir, output_filename)
        writer = tf.python_io.TFRecordWriter(output_file)

        shard_counter = 0
        images_in_shard = np.arange(
            shard_ranges[s], shard_ranges[s+1], dtype=np.int
        )

        for i in images_in_shard:
            image = images[i]

            sequence_example = _to_sequence_example(image, decoder, vocab)
            if sequence_example is not None:
                writer.write(sequence_example.SerializeToString())
                shard_counter += 1
                counter += 1

            if counter % 1000 == 0:
                print(
                    "%s [thread %d]: Processed %d of %d items in thread batch."
                    % (datetime.now(), thread_index, shard_counter, num_images_in_thread)
                )
        writer.close()
        print('%s [thread %d]: Wrote %d image-caption pairs to %d shards.'
              % (datetime.now(), thread_index, counter, num_shards_per_batch))
        sys.stdout.flush()
        shard_counter = 0

    print("%s [thread %d]: Wrote %d image-caption pairs to %d shards." %
          (datetime.now(), thread_index, counter, num_shards_per_batch))
    sys.stdout.flush()


def _process_dataset(name, images, vocab, num_shards):

    images = [
        ImageMetaData(image.image_id, image.filename, [caption])
        for image in images for caption in image.captions
    ]

    random.seed(123456)
    random.shuffle(images)

    num_threads = min(num_shards, FLAGS.num_threads)
    spacing = np.linspace(0, len(images), num_threads+1).astype(np.int)
    ranges = []
    threads = []

    for i in range(len(spacing) - 1):
        ranges.append([spacing[i], spacing[i+1]])

    coord = tf.train.Coordinator()

    decoder = ImageDecoder()

    print('Launching %d threads for spacings: %s' % (num_threads, ranges))

    for thread_index in range(len(ranges)):
        args = (thread_index, ranges, name, images, decoder, vocab, num_shards)
        t = threading.Thread(target=_process_image_files, args=args)
        t.start()
        threads.append(t)

    coord.join(threads)

    print(
        '%s: Finished processing all %d image-caption pairs in data set %s.'
        % (datetime.now(), len(images), name)
    )


def _create_vocab(captions):

    print('Creating vocabulary.')
    counter = collections.Counter()
    for c in captions:
        counter.update(c)
    print('Total words: ', len(counter))

    word_counts = [x for x in counter.items() if x[1] > FLAGS.min_word_count]
    word_counts.sort(key=lambda x: x[1], reverse=True)
    print('Words in vocabulary: ', len(word_counts))

    with tf.gfile.FastGFile(FLAGS.word_counts_output_file, 'w') as f:
        f.write('\n'.join(["%s %d" % (w, c) for w, c in word_counts]))
    print("Wrote vocabulary file: ", FLAGS.word_counts_output_file)

    reverse_vocab = [x[0] for x in word_counts]
    unk_id = len(reverse_vocab)

    vocab_dict = dict([(x, y) for (y, x) in enumerate(reverse_vocab)])
    vocab = Vocabulary(vocab_dict, unk_id)

    return vocab


def _process_caption(caption):
    start_word = bytes(FLAGS.start_word, 'utf-8')
    end_word = bytes(FLAGS.end_word, 'utf-8')
    tokenized_caption = [start_word]
    x = [bytes(word, 'utf-8') for word in nltk.tokenize.word_tokenize(caption.lower())]
    tokenized_caption.extend(x)
    tokenized_caption.append(end_word)

    return tokenized_caption


def _load_and_process_metadata(captions_file, image_dir):
    with tf.gfile.FastGFile(captions_file, 'r') as f:
        caption_data = json.load(f)

    id_to_filename = [(x['id'], x['file_name']) for x in caption_data['images']]

    id_to_captions = {}

    for annotation in caption_data['annotations']:
        image_id = annotation['image_id']
        caption = annotation['caption']

        id_to_captions.setdefault(image_id, [])
        id_to_captions[image_id].append(caption)

    assert len(id_to_filename) == len(id_to_captions)
    assert set([x[0] for x in id_to_filename]) == set(id_to_captions.keys())

    print('Loaded caption metadata for %d images from %s'
          % (len(id_to_filename), captions_file))

    print('Processing captions.')

    image_metadata = []
    num_captions = 0

    for image_id, base_filename in id_to_filename:
        filename = os.path.join(image_dir, base_filename)
        captions = [_process_caption(c) for c in id_to_captions[image_id]]
        image_metadata.append(ImageMetaData(image_id, filename, captions))
        num_captions += len(captions)

    print('Finished processing %d captions for %d images in %s'
          % (num_captions, len(id_to_filename), captions_file))

    return image_metadata


def main(_):
    def _is_valid_num_shards(num_shards):
        return num_shards < FLAGS.num_threads or not num_shards % FLAGS.num_threads

    assert _is_valid_num_shards(FLAGS.train_shards), (
        "Please make the FLAGS.num_threads commensurate with FLAGS.train_shards"
    )
    assert _is_valid_num_shards(FLAGS.val_shards), (
        "Please make the FLAGS.num_threads commensurate with FLAGS.val_shards")
    assert _is_valid_num_shards(FLAGS.test_shards), (
        "Please make the FLAGS.num_threads commensurate with FLAGS.test_shards"
    )

    if not tf.gfile.IsDirectory(FLAGS.output_dir):
        tf.gfile.MakeDirs(FLAGS.output_dir)

    mscoco_train_dataset = _load_and_process_metadata(FLAGS.train_captions_file,
                                                      FLAGS.train_image_dir)

    mscoco_val_dataset = _load_and_process_metadata(FLAGS.val_captions_file,
                                                    FLAGS.val_image_dir)

    train_cutoff = int(0.85 * len(mscoco_val_dataset))
    val_cutoff = int(0.90 * len(mscoco_val_dataset))

    train_dataset = mscoco_train_dataset + mscoco_val_dataset[0:train_cutoff]
    val_dataset = mscoco_val_dataset[train_cutoff:val_cutoff]
    test_dataset = mscoco_val_dataset[val_cutoff:]

    train_captions = [c for image in train_dataset for c in image.captions]
    vocab = _create_vocab(train_captions)

    _process_dataset("train", train_dataset, vocab, FLAGS.train_shards)
    _process_dataset("val", val_dataset, vocab, FLAGS.val_shards)
    _process_dataset("test", test_dataset, vocab, FLAGS.test_shards)


if __name__ == "__main__":
    tf.app.run()
