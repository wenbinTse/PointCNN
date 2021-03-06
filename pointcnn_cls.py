import layers
import tensorflow as tf
from pointcnn import PointCNN


class Net(PointCNN):
    def __init__(self, points, features, is_training, setting):
        PointCNN.__init__(self, points, features, is_training, setting)
        fc_mean = tf.reduce_mean(self.fc_layers[-1], axis=1, keep_dims=True, name='fc_mean')
        self.fc_layers[-1] = tf.cond(is_training, lambda: self.fc_layers[-1], lambda: fc_mean)

        self.logits = layers.dense(
            self.fc_layers[-1],
            setting.num_class,
            'logits',
            is_training,
            with_bn=False,
            activation=None
        )
