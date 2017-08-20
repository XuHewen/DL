import tensorflow as tf


def parse_sequence_example(serialized, image_feature, caption_feature):
    # context_features = {
    #     image_feature: tf.FixedLenFeature([], dtype=tf.string)
    # }
    #
    # sequence_features = {
    #     caption_feature: tf.FixedLenSequenceFeature([], dtype=tf.int64)
    # }
    #
    # context_parsed, sequence_parsed = tf.parse_single_sequence_example(
    #     serialized=serialized,
    #     context_features=context_features,
    #     sequence_features=sequence_features
    # )
    #
    # encoded_image = context_parsed[image_feature],
    # caption = sequence_parsed[caption_feature]
    #
    # return encoded_image, caption
    context, sequence = tf.parse_single_sequence_example(
        serialized,
        context_features={
            image_feature: tf.FixedLenFeature([], dtype=tf.string)
        },
        sequence_features={
            caption_feature: tf.FixedLenSequenceFeature([], dtype=tf.int64),
        })

    encoded_image = context[image_feature]
    caption = sequence[caption_feature]
    return encoded_image, caption


def prefetching_input_data(reader,
                           file_pattern,
                           is_training,
                           batch_size,
                           values_per_shard,
                           input_queue_capacity_factor=16,
                           num_input_reader_threads=1,
                           shard_queue_name='filename_queue',
                           value_queue_name='input_queue'):
    data_files = []
    for pattern in file_pattern.split(','):
        data_files.extend(tf.gfile.Glob(pattern))
    if not data_files:
        tf.logging.fatal('Found no input files matching %s', file_pattern)
    else:
        tf.logging.info('Prefetching values from %d files matching %s',
                        len(data_files), file_pattern)

    if is_training:
        filename_queue = tf.train.string_input_producer(data_files,
                                                        shuffle=True,
                                                        capacity=16,
                                                        name=shard_queue_name)
        min_queue_examples = values_per_shard * input_queue_capacity_factor
        capacity = values_per_shard + 200 * batch_size
        # capacity = 3 * batch_size
        values_queue = tf.RandomShuffleQueue(capacity=capacity,
                                             min_after_dequeue=min_queue_examples,
                                             dtypes=[tf.string],
                                             name='random_' + value_queue_name)

    else:
        filename_queue = tf.train.string_input_producer(data_files,
                                                        shuffle=False,
                                                        capacity=1,
                                                        name=shard_queue_name)
        capacity = values_per_shard + 3 * batch_size
        values_queue = tf.FIFOQueue(capacity=capacity,
                                    dtypes=[tf.string],
                                    name='fifo_' + value_queue_name)

    enqueue_ops = []
    for _ in range(num_input_reader_threads):
        _, value = reader.read(filename_queue)
        enqueue_ops.append(values_queue.enqueue([value]))

    tf.train.queue_runner.add_queue_runner(
        tf.train.queue_runner.QueueRunner(values_queue, enqueue_ops)
    )

    tf.summary.scalar(
        'queue/%s/fraction_of_%d_full' % (values_queue.name, capacity),
        tf.cast(values_queue.size(), tf.float32) * (1. / capacity)
    )

    return values_queue


def batch_with_dynamic_pad(images_and_captions,
                           batch_size,
                           queue_capacity,
                           add_summaries=True):
    enqueue_list = []

    for image, caption in images_and_captions:
        caption_length = tf.shape(caption)[0]
        input_length = tf.expand_dims(tf.subtract(caption_length, 1), 0)

        input_seq = tf.slice(caption, [0], input_length)
        target_seq = tf.slice(caption, [1], input_length)

        indicator = tf.ones(input_length, dtype=tf.int32)
        enqueue_list.append([image, input_seq, target_seq, indicator])

    images, input_seqs, target_seqs, mask = tf.train.batch_join(
        enqueue_list,
        batch_size=batch_size,
        capacity=queue_capacity,
        dynamic_pad=True,
        name='batch_and_pad'
    )

    if add_summaries:
        lengths = tf.add(tf.reduce_sum(mask, 1), 1)
        tf.summary.scalar("caption_length/batch_min", tf.reduce_min(lengths))
        tf.summary.scalar("caption_length/batch_max", tf.reduce_max(lengths))
        tf.summary.scalar("caption_length/batch_mean", tf.reduce_mean(lengths))

    return images, input_seqs, target_seqs, mask
