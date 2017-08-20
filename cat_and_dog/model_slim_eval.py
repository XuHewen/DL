import tensorflow as tf
import math
import model

slim = tf.contrib.slim


flags = tf.flags
FLAGS = flags.FLAGS

flags.DEFINE_string('checkpoint_dir', './logs', 'directory of checkpoints')
flags.DEFINE_integer('num_examples', 7500, 'Number of examples to run')
flags.DEFINE_string('eval_dir', './eval', 'directory for eval')

tf.logging.set_verbosity(tf.logging.INFO)

with tf.Graph().as_default() as g:
    images, labels = model.eval_inputs()
    labels = tf.cast(labels, tf.int64)
    logits = model.inference(images, is_training=False)
    # predictions = tf.nn.in_top_k(logits, labels, 1)
    predictions = tf.argmax(logits, 1)

    saver = tf.train.Saver()

    names_to_values, names_to_updates = slim.metrics.aggregate_metric_map({
        'accuracy': slim.metrics.streaming_accuracy(predictions, labels),
        'precision': slim.metrics.streaming_precision(predictions, labels)
    })

    num_batches = math.ceil(7500/128.0)

    summary_ops = []
    for metric_name, metric_value in names_to_values.items():
        op = tf.summary.scalar(metric_name, metric_value)
        op = tf.Print(op, [metric_value], metric_name)
        summary_ops.append(op)
    print(summary_ops)

    # slim.get_or_create_global_step()
    global_step = tf.Variable(0, name="global_step", trainable=False)
    with tf.Session() as sess:

        output_dir = './eval_slim'
        eval_interval_secs = 600
        slim.evaluation.evaluation_loop(
            '',
            FLAGS.checkpoint_dir,
            output_dir,
            num_evals=num_batches,
            eval_op=list(names_to_updates.values()),
            summary_op=tf.summary.merge(summary_ops),
            eval_interval_secs=eval_interval_secs
        )
