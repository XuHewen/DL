import numpy as np
import tensorflow as tf
from tensorflow.contrib.learn.python.learn.datasets.mnist import read_data_sets

IMAGE_SIZE = 28


def variable_on_cpu(name, shape, initializer):
    with tf.device('/cpu:0'):
        var = tf.get_variable(
            name=name, shape=shape, initializer=initializer, dtype=tf.float32)

    return var


def variable_with_weight_decay(name, shape, stddev=0.02, wd=None):
    var = variable_on_cpu(name, shape,
                          tf.truncated_normal_initializer(
                              stddev=stddev, dtype=tf.float32))
    if wd is not None:
        weight_decay = tf.multiply(tf.nn.l2_loss(var), wd)
        tf.add_to_collection('losses', weight_decay)

    return var


class GAN(object):
    def __init__(self, z_dim, batch_size):
        self.z_dim = z_dim
        self.batch_size = batch_size

    def _generator(self, z, hidden_size):
        with tf.variable_scope('generator') as scope:
            W1 = variable_with_weight_decay('W1', [self.z_dim, hidden_size])
            b1 = variable_on_cpu('b1', [hidden_size],
                                 tf.constant_initializer(0.0))
            W2 = variable_with_weight_decay(
                'W2', [hidden_size, IMAGE_SIZE * IMAGE_SIZE])
            b2 = variable_on_cpu('b2', [IMAGE_SIZE * IMAGE_SIZE],
                                 tf.constant_initializer(0.0))
            h1 = tf.nn.relu(tf.matmul(z, W1) + b1)
            logits = tf.matmul(h1, W2) + b2
            prob = tf.nn.sigmoid(logits, name=scope.name)

            return prob

    def _discriminator(self, image, hidden_size, scope_reuse=False):
        with tf.variable_scope('discriminator') as scope:
            if scope_reuse:
                scope.reuse_variables()

            W1 = variable_with_weight_decay(
                'W1', [IMAGE_SIZE * IMAGE_SIZE, hidden_size])
            b1 = variable_on_cpu('b1', [hidden_size],
                                 tf.constant_initializer(0.0))
            W2 = variable_with_weight_decay('W2', [hidden_size, 1])
            b2 = variable_on_cpu('b2', [1], tf.constant_initializer(0.0))

            h1 = tf.nn.relu(tf.matmul(image, W1) + b1)
            logits = tf.matmul(h1, W2) + b2
            prob = tf.nn.sigmoid(logits)

            return prob, logits

    def _cross_entropy_loss(self, logits, labels):
        xentropy = tf.nn.sigmoid_cross_entropy_with_logits(
            logits=logits, labels=labels)
        xentropy = tf.reduce_mean(xentropy)

        return xentropy

    def _gan_loss(self, logits_real, logits_fake):
        discriminator_loss_real = self._cross_entropy_loss(
            logits_real, tf.ones_like(logits_real))
        discriminator_loss_fake = self._cross_entropy_loss(
            logits_fake, tf.zeros_like(logits_fake))

        self.dis_loss = discriminator_loss_real + discriminator_loss_fake

        self.gen_loss = self._cross_entropy_loss(logits_fake,
                                                 tf.ones_like(logits_fake))

        tf.summary.scalar('losses/discriminator_loss', self.dis_loss)
        tf.summary.scalar('losses/gemerator_loss', self.gen_loss)

    def create_network(self, g_hidden_size, d_hidden_size, learning_rate):
        mnist = read_data_sets('./data', fake_data=False)
        images, _ = mnist.train.next_batch(self.batch_size)

        self.z_vec = tf.placeholder(tf.float32, [self.batch_size, self.z_dim], 'z')

        self.generated = self._generator(self.z_vec, g_hidden_size)

        dis_real_prob, logits_real = self._discriminator(
            images, d_hidden_size, scope_reuse=False)
        dis_fake_prob, logits_fake = self._discriminator(
            self.generated, d_hidden_size, scope_reuse=True)

        # tf.summary.image('real_image',
        #                  tf.reshape(images, [self.batch_size, 28, 28, 1]),
        #                  max_outputs=4)
        tf.summary.image(
            'generated_image', tf.reshape(self.generated,
                                          [self.batch_size, 28, 28, 1]),
            max_outputs=16)
        tf.summary.histogram('z', self.z_vec)

        self._gan_loss(logits_real, logits_fake)

        trainable_variables = tf.trainable_variables()

        self.G_variables = [
            v for v in trainable_variables if v.name.startswith('generator')
        ]
        self.D_variables = [
            v for v in trainable_variables
            if v.name.startswith('discriminator')
        ]

        print(self.D_variables)

        opt = tf.train.AdamOptimizer(learning_rate=learning_rate)

        g_grads = opt.compute_gradients(self.gen_loss,
                                        var_list=self.G_variables)
        d_grads = opt.compute_gradients(self.dis_loss,
                                        var_list=self.D_variables)

        self.G_train_op = opt.apply_gradients(g_grads)
        self.D_train_op = opt.apply_gradients(d_grads)

    def initialize_network(self, logs_dir):
        tf.logging.info('Initialize network...')
        self.logs_dir = logs_dir
        self.sess = tf.Session()
        self.summary_op = tf.summary.merge_all()
        self.saver = tf.train.Saver()
        self.summary_writer = tf.summary.FileWriter(self.logs_dir, self.sess.graph)

        self.sess.run(tf.global_variables_initializer())

        ckpt = tf.train.get_checkpoint_state(self.logs_dir)
        if ckpt and ckpt.model_checkpoint_path:
            self.saver.restore(self.sess, ckpt.model_checkpoint_path)
            print('Model restored...')

        self.coord = tf.train.Coordinator()
        self.threads = tf.train.start_queue_runners(self.sess, self.coord)

    def train_model(self, max_iterations):
        try:
            tf.logging.info("Training model...")
            for itr in range(1, max_iterations):
                batch_z = np.random.uniform(
                    -1.0, 1.0,
                    size=[self.batch_size, self.z_dim]
                ).astype(np.float32)

                feed_dict = {self.z_vec: batch_z}

                self.sess.run(self.D_train_op, feed_dict=feed_dict)
                self.sess.run(self.G_train_op, feed_dict=feed_dict)

                if itr % 1000 == 0:
                    g_loss_val, d_loss_val, summary_str = self.sess.run(
                        [self.gen_loss, self.dis_loss, self.summary_op],
                        feed_dict=feed_dict
                    )

                    print('Step: %d, generator loss: %g, discriminator loss: %g' %
                          (itr, g_loss_val, d_loss_val))
                    self.summary_writer.add_summary(summary_str, itr)

                if itr % 10000 == 0:
                    self.saver.save(self.sess, self.logs_dir + '/model.ckpt',
                                    global_step=itr)
        except tf.errors.OutOfRangeError:
            tf.logging.info('Done training -- epoch limit reached')
        except KeyboardInterrupt:
            tf.logging.info("Ending Training...")
        finally:
            self.coord.request_stop()
            self.coord.join(self.threads)

    def show_image(self):
        import matplotlib.pyplot as plt

        batch_z = np.random.uniform(
            -1.0, 1.0,
            size=[self.batch_size, self.z_dim]
        )
        feed_dict = {self.z_vec: batch_z}

        generated = tf.reshape(self.generated, [self.batch_size, 28, 28, 1])
        images = self.sess.run(generated, feed_dict=feed_dict)

        plt.figure(1)
        for i in range(16):
            plt.subplot(4, 4, i+1)
            plt.imshow(images[i, :, :, 0])
            plt.axis('off')

        plt.show()
