from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np


""" Super Class """
class Optimizer(object):
    """
    This is a template for implementing the classes of optimizers
    """
    def __init__(self, net, lr=1e-4):
        self.net = net  # the model
        self.lr = lr    # learning rate

    """ Make a step and update all parameters """
    def step(self):
        #### FOR RNN / LSTM ####
        if hasattr(self.net, "preprocess") and self.net.preprocess is not None:
            self.update(self.net.preprocess)
        if hasattr(self.net, "rnn") and self.net.rnn is not None:
            self.update(self.net.rnn)
        if hasattr(self.net, "postprocess") and self.net.postprocess is not None:
            self.update(self.net.postprocess)
        
        #### MLP ####
        if not hasattr(self.net, "preprocess") and \
           not hasattr(self.net, "rnn") and \
           not hasattr(self.net, "postprocess"):
            for layer in self.net.layers:
                self.update(layer)


""" Classes """
class SGD(Optimizer):
    """ Some comments """
    def __init__(self, net, lr=1e-4, weight_decay=0.0):
        self.net = net
        self.lr = lr
        self.weight_decay = weight_decay

    def update(self, layer):
        for n, dv in layer.grads.items():
            #############################################################################
            # TODO: Implement the SGD with (optional) Weight Decay                      #
            #############################################################################
            pass
            # theta = v,  dv = DLoss/dw = delta_theta * J(v), lr = Eta, weight_decay = lambda
            v = layer.params[n]
            
            v = v - (self.lr * dv ) - (self.weight_decay * v)
            
            layer.params[n] = v
            #############################################################################
            #                             END OF YOUR CODE                              #
            #############################################################################



class Adam(Optimizer):
    """ Some comments """
    def __init__(self, net, lr=1e-3, beta1=0.9, beta2=0.999, t=0, eps=1e-8, weight_decay=0.0):
        self.net = net
        self.lr = lr
        self.beta1, self.beta2 = beta1, beta2
        self.eps = eps
        self.mt = {}
        self.vt = {}
        self.t = t
        self.weight_decay=weight_decay

    def update(self, layer):
        #############################################################################
        # TODO: Implement the Adam with [optinal] Weight Decay                      #
        #############################################################################
        pass
        # print(self.mt, self.vt)
        for n, dx in layer.grads.items():
            #x is thetaT
            x = layer.params[n]
            #m, v, t at t = t-1
            if n not in self.mt or n not in self.vt:
                self.mt[n] = np.zeros_like(x)
                self.vt[n] = np.zeros_like(x)
            m, v, t= self.mt[n], self.vt[n],self.t

            beta1, beta2 = self.beta1, self.beta2

            #Updating t=> t = t +1 (t goes from t-1 to t)
            t = t+1

            m = beta1 * m + (1.0-beta1) * dx
            v = beta2 * v + (1.0-beta2) * (dx**2)

            mb = m / (1.0 - beta1**float(t))
            vb = v / (1.0 - beta2**float(t))
            
            x_next = x - (self.lr * mb / (np.sqrt(vb) + self.eps)) - (self.weight_decay * x)

            #Updating variables
            self.mt[n], self.vt[n], self.t = m, v, t
            layer.params[n] = x_next
        
        #############################################################################
        #                             END OF YOUR CODE                              #
        #############################################################################
