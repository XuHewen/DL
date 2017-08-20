import tensorflow as tf
import configuration
import im2txt_model

flags = tf.flags
FLAGS = flags.FLAGS

flags.DEFINE_string('input_file_pattern',
                    '/home/xu/store/deep/im2txt/tfrecord/train-?????-of-00256',
                    'File pattern of shared TFRecord input files')
flags.DEFINE_string('vgg16_checkpoint_file', './pre_vgg_ckpt/vgg_16.ckpt',
                    'Path to a pretrained vgg16 model')
flags.DEFINE_string('train_dir', './logs',
                    'Directory for saving and loading model checkpoints')
flags.DEFINE_boolean('train_vgg16', False, 'Whether to train vgg16 submodel variables')
flags.DEFINE_integer('number_of_steps', 1e6, 'number of training steps')
flags.DEFINE_integer('log_every_n_steps', 1,
                     'Frequency at which loss and global step are logged.')

tf.logging.set_verbosity(tf.logging.INFO)


def main(argv=None):
    assert FLAGS.input_file_pattern, "--input_file_pattern is required"
    assert FLAGS.train_dir, '--train_dir is required'

    model_config = configuration.ModelConfig()
    model_config.input_file_pattern = FLAGS.input_file_pattern
    model_config.vgg16_checkpoint_file = FLAGS.vgg16_checkpoint_file

    training_config = configuration.TrainingConfig()

    train_dir = FLAGS.train_dir
    if not tf.gfile.IsDirectory(train_dir):
        tf.logging.info('Creating training directory: %s', train_dir)
        tf.gfile.MakeDirs(train_dir)

    with tf.Graph().as_default() as g:
        model = im2txt_model.Model(model_config,
                                   mode='train',
                                   train_vgg16=FLAGS.train_vgg16)
        model.build_inputs()
        model.build_image_embeddings()
        model.build_seq_embeddings()
        model.build_model()

        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())

            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(coord=coord)

            x = sess.run(model.total_loss)
            print(x)

            coord.request_stop()
            coord.join(threads)


if __name__ == '__main__':
    tf.app.run()
