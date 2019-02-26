# -*- coding: utf_8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf


import argparse
import os
import time

from data_loader import check_dataset_path


def BuildModel(layer_name, attention_nonlinear, attention_type,
               attention_kernel_size):
    # ExtractKeypointDescriptor에 들어갈 _ModelFn을 return한다.
    def _ModelFn(images, normalized_image, reuse):

        if normalized_image:
            image_tensor = images
        else:
            image_tensor = NormalizePixelValues(images)

        # attention scores, features 를 얻기 위함
        model = delf_v1.DelfV1(layer_name)
        _, attention, _, feature_map, _ = model.GetAttentionPrelogit(
            image_tensor,
            attention_nonlinear=attention_nonlinear,
            attention_type=attention_type,
            kernel=[attention_kernel_size, attention_kernel_size],
            training_resnet=False,
            training_attention=False,
            reuse=reuse)
        return attention, feature_map

    return _ModelFn


class DelfInferenceV1(object):
    def __init__(self, model_path=None, config):
        assert tf.train.checkpoint_exists(model_path), "{} is not a tensorflow checkpoint file".format(model_path)


        # config



        sess = tf.Session()

        # attention settings
        layer_name = 'resnet_v1_50/block3'
        attention_nonlinear = 'softplus'
        attention_type = 'use_l2_normalized_feature'
        attention_kernel_size = 1

        # image_scales
        # TODO: read image scale from pbtxt file
        image_scales = tf.constant([0.7072, 1.0])

        # model function
        _model_fn = BuildModel(layer_name, attention_nonlinear, attention_type, attention_kernel_size)

        # graph model



        # weight

        #







if __name__ == '__main__':

    # TODO: Edit help statements
    args = argparse.ArgumentParser()
    args.add_argument('--model_path', type=str, required=True,
                      help='Add trained model.'\
                      'If you did not have any trained model, train from ..script')

    args.add_argument('--epoch', type=int, default=50)
    args.add_argument('--batch_size', type=int, default=64)

    config = args.parse_args()


    # TODO: data path
    data_path = './data'
    check_dataset_path(check_dataset_path)
    m = DelfInferenceV1(data_path, config)
