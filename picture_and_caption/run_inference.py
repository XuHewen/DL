import os

import numpy as np
import tensorflow as tf


import configuration
import inference_wrapper

from inference_utils import caption_generator
from inference_utils import vocabulary


flags = tf.flags
FLAGS = flags.FLAGS


flags.DEFINE_string('checkpoint_path', './logs',
                    'Model checkpoint file'
                    'or directory containing a model checkpoint file')
flags.DEFINE_string('vocab_file',
                    '/home/xu/store/deep/im2txt/annotations/word_counts.txt',
                    'Text file containing the vocabulary')
flags.DEFINE_string('input_files', 'test.jpg',
                    'File pattern of comma-separated list of file patterns'
                    'of image files')

tf.logging.set_verbosity(tf.logging.INFO)


def main(_):
    g = tf.Graph()
    with g.as_default():
        model = inference_wrapper.InferenceWrapper()
        restore_fn = model.build_graph_from_config(configuration.ModelConfig(),
                                                   FLAGS.checkpoint_path)

    g.finalize()

    vocab = vocabulary.Vocabulary(FLAGS.vocab_file)

    filenames = []
    for file_pattern in FLAGS.input_files.split(","):
        filenames.extend(tf.gfile.Glob(file_pattern))
    tf.logging.info("Running caption generation on %d files matching %s",
                    len(filenames), FLAGS.input_files)

    with tf.Session(graph=g) as sess:
        restore_fn(sess)

        generator = caption_generator.CaptionGenerator(model, vocab)

        for filename in filenames:
            with tf.gfile.GFile(filename, 'rb') as f:
                image = f.read()
            captions = generator.beam_search(sess, image)
            print('Captions for image %s: ' % os.path.basename(filename))

            for i, caption in enumerate(captions):

                sentence = [
                    vocab.id_to_word(w) for w in caption.sentence[1:-1]
                ]
                sentence = " ".join(sentence)

                print(" %d) %s (p=%f)" % (i, sentence, np.exp(caption.logprob)))


if __name__ == '__main__':
    tf.app.run()
