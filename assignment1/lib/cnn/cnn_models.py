from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from lib.mlp.fully_conn import *
from lib.mlp.layer_utils import *
from lib.cnn.layer_utils import *


""" Classes """
class TestCNN(Module):
    def __init__(self, dtype=np.float32, seed=None):
        self.net = sequential(
            ########## TODO: ##########
            ConvLayer2D(input_channels=3, kernel_size=3, number_filters=3, stride=1, padding=0, name="conv1", init_scale=.02),
            MaxPoolingLayer(pool_size=2, stride=2, name="maxpool1"),
            flatten(name="flatten"),
            fc(27, 5, 2e-2, name="fc1")
            ########### END ###########
        )


class SmallConvolutionalNetwork(Module):
    def __init__(self, keep_prob=0, dtype=np.float32, seed=None):
        self.net = sequential(
            ########## TODO: ##########
            ConvLayer2D(input_channels=3, kernel_size=5, number_filters=36, stride=2, padding=1, name="conv1", init_scale=.02),
            MaxPoolingLayer(pool_size=2, stride=2, name="maxpool1"),
            flatten(name="flatten"),
            fc(1764, 48, 2e-2, name="fc1"),
            gelu(name="gelu1"),
            fc(48, 20, 2e-2, name="fc2")
            ########### END ###########
        )