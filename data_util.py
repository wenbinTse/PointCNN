import h5py
import os
import numpy as np


def load(filelist_file):
    """
    Load data
    :param filelist_file: the path of a file which stored relative file list
    :return: tuple(points, labels)
    """
    points = []
    labels = []

    folder = os.path.dirname(filelist_file)
    for line in open(filelist_file):
        filename = os.path.basename(line.strip())
        print('files include, ', filename)
        data = h5py.File(os.path.join(folder, filename))
        if 'normal' in data:
            points.append(np.concatenate([data['data'][...], data['normal'][...]], axis=-1).astype(np.float32))
        else:
            points.append(data['data'][...].astype(np.float32))
        labels.append(np.squeeze(data['label'][:]).astype(np.int64))
    return (np.concatenate(points, axis=0),
            np.concatenate(labels, axis=0))


def shuffle(inputs: ([], [])):
    """
    shuffle data
    :param inputs: a tuple whose first element is the array of points, and second is the array of label
    :return: the shuffled data
    """
    assert isinstance(inputs, tuple) and len(inputs) == 2

    indices = np.arange(inputs[0].shape[0])
    np.random.shuffle(indices)

    return inputs[0][indices], inputs[1][indices]


def load_whole_data(filelist_file_train, filelist_file_val):
    data_train, label_train = shuffle(load(filelist_file_train))
    data_val, label_val = load(filelist_file_val)
    return data_train, label_train, data_val, label_val


if __name__ == '__main__':
    filelist_file = './data/modelnet/test_files.txt'
    data = load(filelist_file)
    print(
        type(data), '\n',
        data[0].shape, '\n',
        data[1].shape, '\n',
        data[0][0][0], '\n',
        data[1][0]
    )