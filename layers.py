import tensorflow as tf


def dense(input, output, name, is_training, reuse=None, with_bn=True, activation=tf.nn.elu):
    dense = tf.layers.dense(input, units=output, activation=activation,
                            kernel_initializer=tf.glorot_normal_initializer(),
                            kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=1.0),
                            reuse=reuse, name=name, use_bias=not with_bn)
    return batch_normalization(dense, is_training, name + '_bn', reuse) if with_bn else dense


def batch_normalization(data, is_training, name, reuse=None):
    return tf.layers.batch_normalization(
        data,
        momentum=0.99,
        training=is_training,
        beta_regularizer=tf.contrib.layers.l2_regularizer(scale=1.0),
        gamma_regularizer=tf.contrib.layers.l2_regularizer(scale=1.0),
        reuse=reuse,
        name=name)


def conv2d(input, output, name, is_training, kernel_size,
           reuse=None, with_bn=True, activation=tf.nn.elu):
    conv2d = tf.layers.conv2d(
        input,
        output,
        kernel_size=kernel_size,
        strides=(1, 1),
        padding='VALID',
        activation=activation,
        kernel_initializer=tf.glorot_normal_initializer(),
        kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=1.0),
        reuse=reuse,
        name=name,
        use_bias=not with_bn)
    return batch_normalization(conv2d, is_training, name + '_bn', reuse) if with_bn else conv2d


def depthwise_conv2d(input, depth_multiplier, name, is_training, kernel_size,
                     reuse=None, with_bn=True, activation=tf.nn.elu):
    conv2d = tf.contrib.layers.separable_conv2d(
        input,
        num_outputs=None,
        kernel_size=kernel_size,
        padding='VALID',
        activation_fn=activation,
        depth_multiplier=depth_multiplier,
        weights_initializer=tf.glorot_normal_initializer(),
        weights_regularizer=tf.contrib.layers.l2_regularizer(scale=1.0),
        biases_initializer=None if with_bn else tf.zeros_initializer(),
        biases_regularizer=None if with_bn else tf.contrib.layers.l2_regularizer(scale=1.0),
        reuse=reuse, scope=name)
    return batch_normalization(conv2d, is_training, name + '_bn', reuse) if with_bn else conv2d


def separable_conv2d(input, output, name, is_training, kernel_size, depth_multiplier=1,
                     reuse=None, with_bn=True, activation=tf.nn.elu):
    conv2d = tf.layers.separable_conv2d(
        input,
        output,
        kernel_size=kernel_size,
        strides=(1, 1),
        padding='VALID',
        activation=activation,
        depth_multiplier=depth_multiplier,
        depthwise_initializer=tf.glorot_normal_initializer(),
        pointwise_initializer=tf.glorot_normal_initializer(),
        depthwise_regularizer=tf.contrib.layers.l2_regularizer(scale=1.0),
        pointwise_regularizer=tf.contrib.layers.l2_regularizer(scale=1.0),
        reuse=reuse, name=name,
        use_bias=not with_bn)
    return batch_normalization(conv2d, is_training, name + '_bn', reuse) if with_bn else conv2d