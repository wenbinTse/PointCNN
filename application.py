from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf
import util
import importlib
import sys
import pointcnn_cls as model
import os
import h5py

current_path = os.path.dirname(os.path.abspath(__file__))

sys.path.append(os.path.join(current_path, 'settings'))

setting_name = os.path.join('modelnet_x3_l4_aligned_w_fts')
model_path = os.path.join(current_path, 'pretrained_model', 'iter-28000')

setting = importlib.import_module(setting_name)

is_training = tf.placeholder(tf.bool, name='is_training')
pts_fts = tf.placeholder(tf.float32, shape=(None, 2048, setting.data_dim), name='points')
indices = tf.placeholder(tf.int32, shape=(None, None, 2), name="indices")

pts_fts_sampled = tf.gather_nd(pts_fts, indices=indices, name='pts_fts_sampled')
if setting.data_dim > 3:
    points_sampled, features_sampled = tf.split(pts_fts_sampled,
                                                [3, setting.data_dim - 3],
                                                axis=-1,
                                                name='split_points_features')
    if not setting.use_extra_features:
        features_sampled = None
else:
    points_sampled = pts_fts_sampled
    features_sampled = None

net = model.Net(points_sampled, features_sampled, is_training, setting)
features_layer = net.fc_layers[-1]
logits = net.logits
probs = tf.nn.softmax(logits, name='probs')
predict = tf.nn.top_k(probs, k=5)
k_probs =  predict.values
k_label_id = predict.indices

saver = tf.train.Saver()
sess = tf.Session()
saver.restore(sess, model_path)


def classify(points):
    """
    :param points: points shape is [2, Num_of_point, channels]. The first dim is xyz, while the second is normal
    :param setting_name: the name of setting file
    :return:
    """
    points = np.concatenate([points[0], points[1]], axis=-1)
    points = np.expand_dims(points, 0)
    k_probs_tmp, k_label_id_tmp = sess.run([k_probs, k_label_id], feed_dict={
        is_training: False,
        pts_fts: points,
        indices: util.get_indices(1, 1024, 2018, None),
    })
    print(k_probs_tmp, k_label_id_tmp)
    return np.squeeze(k_probs_tmp), np.squeeze(k_label_id_tmp)


def get_whole_features(h5_file_path='data/modelnet/ply_data_train0.h5', features_file_path='data/modelnet/features'):
    """
    get features of models in h5py file
    :param h5_file_path:
    :return:
    """
    inputs = []
    h5_file = h5py.File(h5_file_path)
    points = h5_file['data']
    normals = h5_file['normal']

    for ps, ns in zip(points, normals):
        tmp = np.concatenate([ps, ns], axis=-1)
        inputs.append(tmp)
    inputs = np.array(inputs)
    print('inputs shape', inputs.shape)

    inputs_size = inputs.shape[0]
    batch_size = 50
    index = 0

    features = []

    while index < inputs_size:
        tmp_input = inputs[index:min(inputs_size, index + batch_size)]
        tmp_feature = get_features(tmp_input)
        features.append(tmp_feature)
        index += tmp_input.shape[0]

    features = np.concatenate(features, axis=0)
    np.save(features_file_path, features)
    print('features shape: ', features.shape)
    print('saved to {}.npy'.format(features_file_path))


def get_features(points_batch):
    features = sess.run(features_layer, feed_dict={
        is_training: False,
        pts_fts: points_batch,
        indices: util.get_indices(points_batch.shape[0], setting.sample_num, 2018, None),
    })
    return np.squeeze(features)


whole_features = None
whole_points = None
def search(points, k=20):
    """
    返回最相似的k个模型（从训练集中提取）
    :param points: 一个点云模型
    :return:
    """
    global whole_features, whole_points

    if whole_features is None:
        h5_file_path = 'data/modelnet/ply_data_train0.h5'
        features_file_path = 'data/modelnet/features.npy'
        h5_file_path = os.path.join(current_path, h5_file_path)
        features_file_path = os.path.join(current_path, features_file_path)

        whole_features = np.load(features_file_path)
        h5_file = h5py.File(h5_file_path)
        whole_points = np.array(h5_file['data'])

    points = np.concatenate([points[0], points[1]], axis=-1)
    points = np.expand_dims(points, 0)

    features = get_features(points)
    features_tile = np.tile(features, (whole_features.shape[0], 1))
    mse = np.sqrt(np.sum(np.power(features_tile - whole_features, 2), axis=1))
    ind = np.argpartition(mse, k)[:k]
    ind = ind[np.argsort(mse[ind])]

    print(type(whole_points))
    print(ind)

    return whole_points[ind]


if __name__ == '__main__':
    # points = np.load('./ui/pointcloud/bed_2_341.npy')
    # classify(points)

    get_whole_features()

    # points = np.load('./ui/pointcloud/bed_2_341.npy')
    # search(points)
