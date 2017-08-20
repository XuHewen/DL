import tensorflow as tf
from model import GAN

flags = tf.flags
FLAGS = tf.flags.FLAGS

flags.DEFINE_integer('batch_size', 128, 'batch size for training')
flags.DEFINE_string('logs_dir', './logs', 'path of logs directory')
flags.DEFINE_integer('z_dim', 1000, 'size of input vector to generator')
flags.DEFINE_float('learning_rate', 1e-3, 'learning rate of Adam Optimizer')
flags.DEFINE_float('iterations', 1e5, 'number of iterations')
flags.DEFINE_integer('g_hidden_size', 128, 'the size of hidden layer')
flags.DEFINE_integer('d_hidden_size', 128, 'the size of hidden layer')
flags.DEFINE_string('mode', 'train', 'train / show')


def main(argv=None):

    model = GAN(FLAGS.z_dim,
                FLAGS.batch_size)

    model.create_network(FLAGS.g_hidden_size, FLAGS.d_hidden_size,
                         FLAGS.learning_rate)

    model.initialize_network(FLAGS.logs_dir)

    if FLAGS.mode == 'train':
        model.train_model(int(1 + FLAGS.iterations))
    else:
        model.show_image()


if __name__ == '__main__':
    tf.app.run()
