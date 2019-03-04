# -*- coding: utf_8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys

from data_loader import *
from train_models import *

dirname = os.path.dirname(os.path.abspath(__file__))

####################[ Download pretrained resnet_v1_50.ckpt ]##################
# This code block is selectable
# You can also download resnet_v1_50.ckpt from
# http://download.tensorflow.org/models/resnet_v1_50_2016_08_28.tar.gz

from google_drive_downloader import GoogleDriveDownloader as gdd

if not os.path.exists("resnet_v1_50.ckpt"):
    ckpt_id = "1EorhNWDmU1uILq3qetrdz3-fieH3QFtK"
    gdd.download_file_from_google_drive(
        file_id=ckpt_id,
        dest_path=os.path.join(dirname, 'resnet_v1_50.ckpt'),
        unzip=False)
print("resnet_v1_50.ckpt download is completed")
###############################################################################

sys.path.insert(0, os.path.join(dirname, "models/research/delf/delf"))
sys.path.insert(1, os.path.join(dirname, "models/research/delf"))
sys.path.insert(2, os.path.join(dirname, "models/research/slim"))
sys.path.insert(3, os.path.join(dirname, "models/research"))

from python import delf_v1
from nets import resnet_v1

import tensorflow as tf

slim = tf.contrib.slim

@time_checker
def build_resnet(images, num_classes, last_layer, is_training=True,
                 reuse=None):
    model = delf_v1.DelfV1()

    with slim.arg_scope(resnet_v1.resnet_arg_scope(use_batch_norm=True)):
        net, end_points = model.GetResnet50Subnetwork(
            images, global_pool=True, is_training=is_training, reuse=reuse)

    with slim.arg_scope(
            resnet_v1.resnet_arg_scope(
                weight_decay=0.0001, batch_norm_scale=True)):
        with slim.arg_scope([slim.batch_norm], is_training=True):
            feature_map = end_points[last_layer]

            # 첫 번째 conv2d with kernel_size 1 + reduce_mean + expand_dims = Global Average Pooling
            feature_map = slim.conv2d(
                feature_map,
                512,
                1,
                rate=1,
                activation_fn=tf.nn.relu,
                scope='conv1')
            feature_map = tf.reduce_mean(feature_map, [1, 2])
            feature_map = tf.expand_dims(tf.expand_dims(feature_map, 1), 2)

            # 두 번째 conv2d with kernel_size 1 + squeeze = Fully connected Softmax layer
            logits = slim.conv2d(
                feature_map,
                num_classes, [1, 1],
                activation_fn=None,
                normalizer_fn=None,
                scope='logits')
            logits = tf.squeeze(logits, [1, 2], name='spatial_squeeze')
    return logits, feature_map

_SUPPORTED_TRAINING_STEP = ['resnet_default_train', 'resnet_attention_train']


class Config():
    def __init__(self):
        self.batch_size = 64
        self.num_preprocess_threads = 8
        self.nb_epoch = 10
        self.fc_learning_rate = 0.0001
        self.fc_epoch = 5
        self.conv_learning_rate = 0.0001
        self.img_shape = (224, 224, 3)
        self.restore_file = "resnet_v1_50.ckpt"
        self.last_layer = 'resnet_v1_50/block3'
        self.data_path = None
        self.save_name = 'local_ckpt/resnet_tune'
        self.dash_size = 80
        self.sess = None
        self.images = None
        self.labels = None
        self.num_train_batches = None
        self.num_val_batches = None
        self.images_holder = None
        self.logits = None
        self.feature_map = None
        self.num_classes = None
        self.train_step = _SUPPORTED_TRAINING_STEP[0]


class DelfTrainerV1(object):
    # delf trainer initializer
    def __init__(self, config):
        self.config = config
        train_data_path = config.data_path + '/train'
        config.sess = tf.Session()
        config.images, config.labels, config.num_train_batches, \
            config.num_val_batches = pipe_data(config, train_data_path)
        batch_shape = (None, *config.img_shape)
        config.images_holder = tf.placeholder(shape=batch_shape,
                                              dtype=tf.float32)
        if config.train_step is 'resnet_default_train':
            config.logits, config.feature_map = self._build_resnet_graph()
        elif config.train_step is 'resnet_attention_train':
            # _build_attention_graph
            pass
        else:
            raise Exception("You should specify correct train step")

    # build resnet model to fine-tune
    def _build_resnet_graph(self):
        config = self.config
        logits, feauture_map = build_resnet(config.images_holder,
                                            config.num_classes,
                                            config.last_layer)
        return logits, feauture_map

    # build attention model to attention study
    def _build_attention_graph(self):
        return None

    # execute training with the built graph
    def run(self):
        config = self.config
        if config.train_step == "resnet_default_train":
            train_resnet(config)
        elif config.train_step == "resnet_attention_train":
            pass



