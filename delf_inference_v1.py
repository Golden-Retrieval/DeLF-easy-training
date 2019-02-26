# -*- coding: utf_8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

class DelfInferenceV1(object):
    def __init__(self, model_path=None):
        if model_path is None :
            # error
            assert model_path, "ERROR: You did not pass the model_path argument."

        sess = tf.Session()