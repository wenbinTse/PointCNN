"""
Convert the whole test.h5py file to single numpy files
"""

import h5py
import os
import numpy as np
import json

current_path = os.path.dirname(os.path.abspath(__file__))
data_path = os.path.join(current_path, os.pardir, 'data', 'modelnet')

file = h5py.File(os.path.join(data_path, 'ply_data_test1.h5'))

with open(os.path.join(data_path, 'ply_data_test_1_id2file.json')) as f:
    labels = json.load(f)
    print('load data from ply_data_test_1_id2file.json')

points = file['data']
label_ids = file['label']
normals = file['normal']

model2label_id = {}
label_id2label = {}
model_folder = os.path.join(current_path, 'pointcloud')

if not os.path.exists(model_folder):
    os.makedirs(model_folder)

for index, ps in enumerate(points):
    label_id = label_ids[index][0] # label id
    label = labels[index].split('/')[0] # label name
    label_id2label[label_id] = label

    model = np.array([ps, normals[index]])

    np.save(model_folder + '/{}_{}_{}'.format(label, label_id, index), model)


label_id2label = dict([(str(k), label_id2label[k]) for k in sorted(label_id2label.keys())])

with open('label_id2label.json', 'w') as f:
    json.dump(label_id2label, f, indent=4)

print('save models to {}'.format(model_folder))