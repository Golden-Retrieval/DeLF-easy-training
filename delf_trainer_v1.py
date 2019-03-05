# -*- coding: utf_8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
import tensorflow as tf

from data_loader import *
from train_models import *

dirname = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(dirname, "models/research/delf/delf"))
sys.path.insert(1, os.path.join(dirname, "models/research/delf"))
sys.path.insert(2, os.path.join(dirname, "models/research/slim"))
sys.path.insert(3, os.path.join(dirname, "models/research"))

from python import delf_v1
from nets import resnet_v1

slim = tf.contrib.slim

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

_SUPPORTED_TRAINING_STEP = ['resnet_finetune', 'att_learning']
_SUPPORTED_ATTENTION_TYPES = [
    'use_l2_normalized_feature', 'use_default_input_feature'
]
_SUPPORTED_CHECKPOINT_TYPE = ['resnet_ckpt', 'attention_ckpt']


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


def build_attention_model(images, num_classes, sess, is_training=True,
                          reuse=False):
    use_batch_norm = True
    weight_decay = 0.0001
    attention_nonlinear = 'softplus'
    attention_type = _SUPPORTED_ATTENTION_TYPES[0]
    training_resnet = False
    training_attention = True
    kernel = 1

    model = delf_v1.DelfV1()
    with slim.arg_scope(
            resnet_v1.resnet_arg_scope(use_batch_norm=use_batch_norm)):
        attention_feat, attention_prob, attention_score, feature_map, _ = (
            model.GetAttentionPrelogit(
                images,
                weight_decay,
                attention_nonlinear=attention_nonlinear,
                attention_type=attention_type,
                kernel=kernel,
                training_resnet=training_resnet,
                training_attention=training_attention,
                reuse=reuse))

    with slim.arg_scope(
            resnet_v1.resnet_arg_scope(
                weight_decay=weight_decay, batch_norm_scale=True)):
        with slim.arg_scope([slim.batch_norm], is_training=training_attention):
            with tf.variable_scope(
                    "attention_block", values=[attention_feat], reuse=reuse):
                logits = slim.conv2d(
                    attention_feat,
                    num_classes, [1, 1],
                    activation_fn=None,
                    normalizer_fn=None,
                    scope='logits')
                logits = tf.squeeze(logits, [1, 2], name='spatial_squeeze')
    return logits, attention_prob, feature_map


def restore_weight(sess, ckpt_type, ckpt_path='resnet_v1_50.ckpt'):
    """
    If you have "abc.meta, abc.index, abc.data" then just give "abc" to ckpt_path.
    """
    #########################[ Initialize all variable ]#############################
    global_init = tf.global_variables_initializer()
    sess.run(global_init)

    ############## [ restore variable from exist checkpoint] ##############
    if ckpt_type == _SUPPORTED_CHECKPOINT_TYPE[0]:
        restore_var = [v for v in tf.global_variables() if 'resnet' in v.name]
    elif ckpt_type == _SUPPORTED_CHECKPOINT_TYPE[1]:
        restore_var = [v for v in tf.global_variables() if
                       ('resnet' in v.name) or ('attention_block' in v.name)]
    else:
        raise Exception(
            "You should fill valid config.ckpt_type: Check delf_train.py _SUPPORTED_CHECKPOINT_TYPE")
    saver = tf.train.Saver(restore_var)
    saver.restore(sess, ckpt_path)
    print("model weights completed")
    return None


@time_checker
def build_attention(images, num_classes, sess, is_training=True, reuse=False):
    use_batch_norm = True
    weight_decay = 0.0001
    attention_nonlinear = 'softplus'
    attention_type = _SUPPORTED_ATTENTION_TYPES[0]
    training_resnet = False
    training_attention = True
    kernel = 1

    model = delf_v1.DelfV1()
    with slim.arg_scope(
            resnet_v1.resnet_arg_scope(use_batch_norm=use_batch_norm)):
        attention_feat, attention_prob, attention_score, feature_map, _ = (
            model.GetAttentionPrelogit(
                images,
                weight_decay,
                attention_nonlinear=attention_nonlinear,
                attention_type=attention_type,
                kernel=kernel,
                training_resnet=training_resnet,
                training_attention=training_attention,
                reuse=reuse))

    with slim.arg_scope(
            resnet_v1.resnet_arg_scope(
                weight_decay=weight_decay, batch_norm_scale=True)):
        with slim.arg_scope([slim.batch_norm], is_training=training_attention):
            with tf.variable_scope(
                    "attention_block", values=[attention_feat], reuse=reuse):
                logits = slim.conv2d(
                    attention_feat,
                    num_classes, [1, 1],
                    activation_fn=None,
                    normalizer_fn=None,
                    scope='logits')
                logits = tf.squeeze(logits, [1, 2], name='spatial_squeeze')
    return logits, attention_prob, feature_map


class Config():
    def __init__(self):
        self.batch_size = 64
        self.num_preprocess_threads = 8
        self.nb_epoch = 10
        self.fc_learning_rate = 0.0001
        self.fc_epoch = 5
        self.conv_learning_rate = 0.0001
        self.att_learning_rate = 0.0001
        self.img_shape = (224, 224, 3)
        self.restore_file = "resnet_v1_50.ckpt"
        self.last_layer = 'resnet_v1_50/block3'
        self.data_path = "Fill it"
        self.train_data_path = "Don't Fill it"
        self.save_name = 'local_ckpt/resnet_tune'
        self.dash_size = 80
        self.sess = "Don't Fill it"
        self.images = "Don't Fill it"
        self.labels = "Don't Fill it"
        self.num_train_batches = "Don't Fill it"
        self.num_val_batches = "Don't Fill it"
        self.images_holder = "Don't Fill it"
        self.logits = "Don't Fill it"
        self.feature_map = "Don't Fill it"
        self.num_classes = "Don't Fill it"
        self.ckpt_type = "Fill it"
        self.train_step = _SUPPORTED_TRAINING_STEP[0]


class DelfTrainerV1(object):
    # delf trainer initializer
    def __init__(self, config):

        ###################[ config, sess, pipeline ]#######
        self.config = config
        config.train_data_path = config.data_path + "/train"
        config.sess = tf.Session()
        self._pipeline_data()

        ###################[ build model ]##################
        if config.train_step is 'resnet_finetune':
            self._build_resnet_graph()
        elif config.train_step is 'att_learning':
            self._build_attention_graph()
        else:
            raise Exception("You should specify correct train step")

        ###################[ restore checkpoint ]############
        restore_weight(config.sess, config.ckpt_type, config.restore_file)

    def _pipeline_data(self):
        config = self.config
        config.images, config.labels, config.num_train_batches, \
        config.num_val_batches = pipe_data(self.config, config.train_data_path)
        batch_shape = (None, *config.img_shape)
        config.images_holder = tf.placeholder(shape=batch_shape,
                                              dtype=tf.float32)

    # build resnet model to fine-tune
    def _build_resnet_graph(self):
        config = self.config
        config.logits, config.feature_map = build_resnet(config.images_holder,
                                                         config.num_classes,
                                                         config.last_layer)
        return None

        # build attention model to attention study

    def _build_attention_graph(self):
        config = self.config
        config.logits, config.attention_prob, feature_map = build_attention(
            config.images_holder, config.num_classes, config.sess)
        return None

    # execute training with the built graph
    def run(self):
        config = self.config
        if config.train_step == "resnet_finetune":
            train_resnet(config)
        elif config.train_step == "att_learning":
            train_att(config)




