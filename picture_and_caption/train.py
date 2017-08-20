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
flags.DEFINE_integer('number_of_steps', 1000000, 'number of training steps')
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

    g = tf.Graph()

    with g.as_default():
        model = im2txt_model.Model(model_config,
                                   mode='train',
                                   train_vgg16=FLAGS.train_vgg16)
        model.build()

        learning_rate_decay_fn = None
        if FLAGS.train_vgg16:
            learning_rate = tf.constant(
                training_config.train_vgg16_learning_rate
            )
        else:
            learning_rate = tf.constant(training_config.initial_learning_rate)
            if training_config.learning_rate_decay_factor > 0:
                num_batches_per_epoch = (training_config.num_examples_per_epoch
                                         / model_config.batch_size)
                decay_steps = int(num_batches_per_epoch *
                                  training_config.num_epochs_per_decay)

                def _learning_rate_decay_fn(learning_rate, global_step):
                    return tf.train.exponential_decay(
                        learning_rate,
                        global_step,
                        decay_steps=decay_steps,
                        decay_rate=training_config.learning_rate_decay_factor,
                        staircase=True
                    )
                learning_rate_decay_fn = _learning_rate_decay_fn

        train_op = tf.contrib.layers.optimize_loss(
            loss=model.total_loss,
            global_step=model.global_step,
            learning_rate=learning_rate,
            optimizer=training_config.optimizer,
            clip_gradients=training_config.clip_gradients,
            learning_rate_decay_fn=learning_rate_decay_fn
        )

        saver = tf.train.Saver(max_to_keep=training_config.max_checkpoints_to_keep)

    tf.contrib.slim.learning.train(
        train_op,
        train_dir,
        log_every_n_steps=FLAGS.log_every_n_steps,
        graph=g,
        global_step=model.global_step,
        number_of_steps=FLAGS.number_of_steps,
        init_fn=model.init_fn,
        saver=saver
    )


if __name__ == '__main__':
    tf.app.run()
