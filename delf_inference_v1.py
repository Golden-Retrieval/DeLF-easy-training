# -*- coding: utf_8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
import tensorflow as tf

from data_loader import load_dataset, _parse_function
from train_models import train_model

dirname = os.path.dirname(os.path.abspath(__file__))

############################[ Local test ]#############################
dirname = "/home/soma03/projects/ai/final"
###############################################################################
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

# import for inference
import argparse
import faiss
from google.protobuf import text_format
from delf import delf_config_pb2
from python.feature_extractor import *
from tqdm import trange
import time
from sklearn.externals import joblib
import numpy as np
from collections import Counter

num_preprocess_threads = 32


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
    def __init__(self, model_path=None):
        # 1.1
        assert tf.train.checkpoint_exists(model_path), "{} is not a tensorflow checkpoint file".format(model_path)        
        ckpt = tf.train.latest_checkpoint(model_path)
        
        # 1.3
        dim = 40          # dimension
        n_subq = 8        # number of sub-quantizers
        n_centroids = 32  # number of centroids for each sub-vector
        n_bits = 5        # number of bits for each sub-vector
        n_probe = 3       # number of voronoi cell to explore
        coarse_quantizer = faiss.IndexFlatL2(dim)
        pq = faiss.IndexIVFPQ(coarse_quantizer, dim, n_centroids, n_subq, n_bits) 
        pq.nprobe = n_probe
        self.pq = pq
        
        # 1.4 build graph
        
        # 1.4.1 define placeholder
        self.images_holder = tf.placeholder(shape=(224, 224, 3), dtype=tf.float32)
        self.labels_holder = tf.placeholder(shape=(), dtype=tf.int32) # not used in BuildModel
        num_classes = 137 # TODO: edit class to get_class_num function
        
        # 1.4.2 build model
        layer_name = 'resnet_v1_50/block3'
        attention_nonlinear = 'softplus'
        attention_type = 'use_l2_normalized_feature'
        attention_kernel_size = 1

        # Parse DelfConfig proto.
        delf_config = delf_config_pb2.DelfConfig()
        delf_config_path = 'delf_config.pbtxt'
        with tf.gfile.FastGFile(delf_config_path, 'r') as f:
            text_format.Merge(f.read(), delf_config)

        # image_scales
        # TODO: collect all configs into delf_config.pbtxt file
        #image_scales = tf.constant([0.7072, 1.0])
        image_scales = tf.constant([0.25, 0.3536, 0.5000, 0.7072, 1.0])
        max_feature_num = 500

        print('\n\n--_model_fn')
        # model function
        _model_fn = BuildModel(layer_name, attention_nonlinear, attention_type, attention_kernel_size)

        print('\n\n--ExtractKeypointDescriptor')
        # ExtractKeypointDescriptor from delf class
        boxes, feature_scales, features, scores = (
            ExtractKeypointDescriptor(
                self.images_holder,
                layer_name='resnet_v1_50/block3',
                image_scales=image_scales,
                iou=1.0,
                max_feature_num=max_feature_num,
                abs_thres=1.5,
                model_fn=_model_fn))
        
        print('\n\n--DelfFeaturePostProcessing')
        # get end nodes
        raw_descriptors = features
        self.locations_node, self.descriptors_node = DelfFeaturePostProcessing(
            boxes, raw_descriptors, delf_config)
        
        print('\n\n--sess')
        # 1.2
        self.sess = tf.Session(config=tf.ConfigProto(
      allow_soft_placement=True, log_device_placement=True))
        
        # 1.4.3 load weights
        
        print('\n\n--init')
        # global initializer
        print('----global_variables_initializer')
        init_op = tf.global_variables_initializer()    
        print('----run')
        self.sess.run(init_op)
        print('----run_')
        
        print('\n\n--restore')
        # restore
        restore_var = [v for v in tf.global_variables() if 'resnet' in v.name]
        saver = tf.train.Saver(restore_var)
        saver.restore(self.sess, ckpt)
        #saver.restore(self.sess, "resnet_v1_50.ckpt")
        
        print('weight loaded')
        
    # 2
    def attach_db_from_path(self, db_path, ignore_cache=False, cache_path='result_cache.joblib', filename_path='filename_path.joblib'):
        
        # 2.1
        self.db_image_paths, self.db_image_labels = load_dataset(db_path)
        # ignore cache loading & execute inference
        if ignore_cache or not os.path.exists(cache_path):
            
            result = self.infer_image_to_des(self.db_image_paths, self.db_image_labels) # result['locations'], result['descriptors']
            # cache save
            with open(cache_path, 'wb') as f:
                joblib.dump(result, f)
#             with open(filename_path, 'wb') as f:
#                 joblib.dump(self.db_image_paths, f)
                
        # exist cache file        
        else:
            print("no inference on db")
            with open(cache_path, 'rb') as f:
                result = joblib.load(f)
#             with open(filename_path, 'rb') as f:
#                 self.db_image_paths = joblib.load(f)
            
        # 2.2
        self.des_from_img, self.img_from_des = make_index_table(result['descriptors'])

        # 2.3
        descriptors_np = np.concatenate(np.asarray(result['descriptors']), axis=0)
        if not self.pq.is_trained:
            self.pq.train(descriptors_np)
        self.pq.add(descriptors_np)

        
    # inference the image list from path to the list of descriptors 
    def infer_image_to_des(self, image_paths, image_labels):
            
        # image_paths, image_labels = list_images(dataset_path)
        
        # get number of classes
        num_classes = len(set(image_paths))
        
        image_dataset = tf.data.Dataset.from_tensor_slices((image_paths, image_labels))
        image_dataset = image_dataset.shuffle(buffer_size=len(image_paths))
        image_dataset = image_dataset.map(_parse_function, num_parallel_calls=num_preprocess_threads)
        
        iterator = image_dataset.make_initializable_iterator()
        iterator_init = iterator.make_initializer(image_dataset)

        # generate input data
        images, labels = iterator.get_next()
        self.sess.run(iterator_init)
        
        locations_list = []
        descriptors_list = []
        
        t0 = time.time()
        t0_small = time.time()
        total_index = 0
        for index in trange(len(image_paths)):
            
            # get locations & descriptors
            input_images, input_labels = self.sess.run([images, labels])
            feed_dict={self.images_holder: input_images, self.labels_holder: input_labels}
            locations_out, descriptors_out = self.sess.run([self.locations_node, self.descriptors_node], feed_dict=feed_dict)
            locations_list.append(locations_out)
            descriptors_list.append(descriptors_out)
            
            if index != 0 and index % 1000 == 0:
                t1_small = time.time()
                print(index, 'th extracting took {:.3f} s'.format((t1_small-t0_small)))
                print('location: ', locations_out.shape, ' descriptor: ', descriptors_out.shape)
                t0_small = time.time()    
        result = {
            'descriptors' : descriptors_list,
            'locations' : locations_list,
        }
        t1 = time.time()
        print('='*30)
        print(len(image_paths), ' extracting took {:.3f} s'.format((t1-t0)))
        

        return result
    
    # 3. search query  
    def search_from_path(self, query_path, top_k=5, verification=True):       
        # 3.1 inference query images to descriptors (infer_image_to_des)
        self.query_image_paths, self.query_image_labels = load_dataset(query_path, no_label=True)
        
        
        query_result = self.infer_image_to_des(self.query_image_paths, self.query_image_labels) # result['locations'], result['descriptors']
        query_des_np = np.concatenate(np.asarray(query_result['descriptors']), axis=0)
        # index table for query set
        self.query_des_from_img, self.query_img_from_des = make_index_table(query_result['descriptors'])
        query_img_idx = list(self.query_des_from_img.keys())
        
        
        # 3.2 pq search
        k = 60  # k nearest neighber
        _, query_des2desList = self.pq.search(query_des_np, k) 
        
        # 3.3 find similar image list by frequency score (get_similar_img(mode='frequency', searched_des))

        query_des2imgList = {}


        # travel query images' inferenced descriptors
        for img_i, des_list in enumerate(query_des2desList):
            # map inferenced descirptors to their parents' image index
            query_des2imgList[img_i] = [self.img_from_des[des_i] for des_i in des_list]

        """
            query_des2imgList = {
                image_index: [list of image index of each descriptor]}
        """
        
        query_img2imgFreq = self.get_similar_img(query_des2imgList, top_k)
        
        # TODO: 3.4 verification by ransac (rerank)
        
        # 3.5 index to image path
        
        for query_i in query_img2imgFreq:
            top_k_img_i_list = query_img2imgFreq[query_i]['index']
            top_k_img_path = [self.db_image_paths[img_i] for img_i in top_k_img_i_list]
            query_img2imgFreq[query_i]['path'] = top_k_img_path
            
        self.result = query_img2imgFreq
        return query_img2imgFreq
            
        
    def print_result(self):
        for i in self.result:
            print('{}th query ({}): '.format(i, self.query_image_paths[i]))
            indices = self.result[i]['index']
            for i, db_index in enumerate(indices):
                print('  top {}: {}'.format(i, self.db_image_paths[db_index]))
    
    def get_similar_img(self, query_des2imgList, top_k):
        
        query_img2imgFreq = {}

        # travel each query image's descriptor list
        for img_i in self.query_des_from_img:
            # img_i : query image index
            # query_des_from_img[img_i] : list of descrptors' indices per image index
        #     print(img_i, query_des_from_img[img_i])

            # aggregate all searched descriptors to one list per one image
            all_searched_des = []
            # travel 
            for des_i in self.query_des_from_img[img_i]:
                all_searched_des.extend(query_des2imgList[des_i])

            imgFreq = Counter(all_searched_des).most_common()
            imgFreq = imgFreq[:top_k]
            index, freq = list(zip(*imgFreq))
            query_img2imgFreq[img_i] = {'index': index, 'freq':freq}
            
        result = query_img2imgFreq
        return query_img2imgFreq
        
        

        
def flatten(x):
    return list(itertools.chain.from_iterable(x))
        
def ensure_list(path):
    if isinstance(path, list):
        return path
    else:
        return [path]

        
def make_index_table(descriptors_list):    
    des_from_img = {}
    img_from_des = {}
    cnt = 0
    for i_img, des_list in enumerate(descriptors_list):
        i_des_range = range(cnt, cnt+len(des_list))
        des_from_img[i_img] = list(i_des_range)
        for i_des in i_des_range:
            img_from_des[i_des] = i_img

        # print(i_img, list(i_des_range))
        cnt+=len(des_list)

    return des_from_img, img_from_des







if __name__ == '__main__':

    # TODO: Edit help statements
    args = argparse.ArgumentParser()
    args.add_argument('--model_path', type=str, default='/home/soma03/projects/ai/final/local_ckpt',
                      help='Add trained model.'\
                      'If you did not have any trained model, train from ..script')
    args.add_argument('--db_path', type=str, default='/home/soma03/projects/data/landmark/cleand/image20/local/train/train_data')
    args.add_argument('--query_path', type=str, default='/home/soma03/projects/data/landmark/cleand/image20/local/train/train_data/31/')

    args.add_argument('--epoch', type=int, default=50)
    args.add_argument('--batch_size', type=int, default=64)

    config = args.parse_args()
    
    # 1. initialize delf_model instance 
    # 1.1 check model path
    # 1.2 get session
    # 1.3 initialize faiss object
    # TODO: 1.4 build graph
    print('\n\n1')
    print('='*30)
    delf_model = DelfInferenceV1(model_path=config.model_path)

    # 2.attach db image path to delf_model instance
    # 2.1 inference db images to descriptors (infer_image_to_des)
    # 2.2 make indices dicts, img_from_des and des_from_img
    # 2.3 pq train & add
    print('\n\n2')
    print('='*30)

    delf_model.attach_db_from_path(config.db_path)

    # 3. search query 
    # 3.1 inference query images to descriptors (infer_image_to_des)
    # 3.2 pq search
    # 3.3 find similar image list by frequency score (get_similar_img(mode='frequency', searched_des))
    # 3.4 verification by ransac (rerank)
    print('\n\n3')
    print('='*30)

    delf_model.search_from_path(config.query_path, top_k=5, verification=True)

    # print
    print('\n\n4')
    print('='*30)
    delf_model.print_result()
    
    print('\n\n5')
    print('='*30)