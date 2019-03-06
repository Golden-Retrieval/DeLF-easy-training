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
        # 1.1
        # assert tf.train.checkpoint_exists(model_path), "{} is not a tensorflow checkpoint file".format(model_path)        
        
        # 1.3
        dim = 40          # dimension
        n_subq = 8        # number of sub-quantizers
        n_centroids = 32  # number of centroids for each sub-vector
        n_bits = 5        # number of bits for each sub-vector
        n_probe = 3       # number of Nearest Neighbors
        coarse_quantizer = faiss.IndexFlatL2(dim)
        pq = faiss.IndexIVFPQ(coarse_quantizer, dim, n_centroids, n_subq, n_bits) 
        pq.nprobe = n_probe
        self.pq = pq
        
        # 1.4 build graph
        
        # 1.4.1 define placeholder
        image_scales = tf.constant([0.25, 0.3536, 0.5000, 0.7072, 1.0])
        images_holder = tf.placeholder(shape=(224, 224, 3), dtype=tf.float32)
        labels_holder = tf.placeholder(shape=(), dtype=tf.int32) # not used in BuildModel
        num_classes = 137 # TODO: edit class to get_class_num function
        
        # 1.4.2 build model
        layer_name = 'resnet_v1_50/block3'
        attention_nonlinear = 'softplus'
        attention_type = 'use_l2_normalized_feature'
        attention_kernel_size = 1

        # Parse DelfConfig proto.
        from delf import delf_config_pb2
        delf_config = delf_config_pb2.DelfConfig()
        delf_config_path = 'delf_config.pbtxt'
        with tf.gfile.FastGFile(delf_config_path, 'r') as f:
            text_format.Merge(f.read(), delf_config)

        # image_scales
        # TODO: read image scale from pbtxt file
        image_scales = tf.constant([0.7072, 1.0])

        # model function
        _model_fn = BuildModel(layer_name, attention_nonlinear, attention_type, attention_kernel_size)

        # ExtractKeypointDescriptor from delf class
        boxes, feature_scales, features, scores = (
            ExtractKeypointDescriptor(
                images_holder,
                layer_name='resnet_v1_50/block3',
                image_scales=image_scales,
                iou=1.0,
                max_feature_num=max_feature_num,
                abs_thres=1.5,
                model_fn=_model_fn))
        
        # get end nodes
        raw_descriptors = features
        locations, descriptors = DelfFeaturePostProcessing(
            boxes, raw_descriptors, delf_config)
        
        # 1.2
        self.sess = tf.Session()
        
        # 1.4.3 load weights
        
        # global initializer
        init_op = tf.global_variables_initializer()    
        self.sess.run(init_op)
        
        # restore
        restore_var = [v for v in tf.global_variables() if 'resnet' in v.name]
        saver = tf.train.Saver(restore_var)
        saver.restore(sess, "resnet_v1_50.ckpt")
        print('weight loaded')
        
    # 2
    def attach_db_from_path(self, db_path):
        # 2.0
        db_des_list = image2des(db_path)
        # 2.1
        
    # inference the image list from path to the list of descriptors 
    def image2des(self, image_path):
        image_path_list = ensure_list(image_path)
        
        
def ensure_list(path):
    if isinstance(path, list):
        return path
    else:
        return [path]

        









if __name__ == '__main__':

    # TODO: Edit help statements
    args = argparse.ArgumentParser()
    args.add_argument('--model_path', type=str, required=True,
                      help='Add trained model.'\
                      'If you did not have any trained model, train from ..script')

    args.add_argument('--epoch', type=int, default=50)
    args.add_argument('--batch_size', type=int, default=64)

    config = args.parse_args()

    
    # TODO: data path check
    db_path = './data'
    check_dataset_path(check_dataset_path)
    
    
    # 1. initialize delf_model instance 
    # 1.1 check model path
    # 1.2 get session
    # 1.3 initialize faiss object
    # TODO: 1.4 build graph
    delf_model = DelfInferenceV1(config.model_path)

    # 2.attach db image path to delf_model instance
    delf_model.attach_db_from_path(db_path)
    
  