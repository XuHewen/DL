import tensorflow as tf
import yaml

slim = tf.contrib.slim


def variable_on_cpu(name, shape, initializer):
    with tf.device('/cpu:0'):
        var = tf.get_variable(name=name,
                              shape=shape,
                              initializer=initializer,
                              dtype=tf.float32)

    return var


def variable_with_weight_decay(name, shape, stddev=0.02, wd=None):
    var = variable_on_cpu(name,
                          shape,
                          tf.truncated_normal_initializer(stddev=stddev,
                                                          dtype=tf.float32))
    if wd is not None:
        weight_decay = tf.multiply(tf.nn.l2_loss(var), wd)
        tf.add_to_collection('losses', weight_decay)

    return var


class Flag(object):
    def __init__(self, **entries):
        self.__dict__.update(entries)


def read_conf_file(conf_file):
    with open(conf_file) as f:
        FLAGS = Flag(**yaml.load(f))
    return FLAGS


def _get_init_fn(FLAGS):
    tf.logging.info('Use pretrained model %s' % FLAGS.loss_model_file)

    exclusions = []
    if FLAGS.checkpoint_exclude_scopes:
        exclusions = [scope.strip()
                      for scope in FLAGS.checkpoint_exclude_scopes.split(',')]

    variables_to_restore = []

    for var in slim.get_model_variables():
        excluded = False
        for exclusion in exclusions:
            if var.op.name.startswith(exclusion):
                excluded = True
                break
        if not excluded:
            variables_to_restore.append(var)

    return slim.assign_from_checkpoint_fn(FLAGS.loss_model_file,
                                          variables_to_restore,
                                          ignore_missing_vars=True)
