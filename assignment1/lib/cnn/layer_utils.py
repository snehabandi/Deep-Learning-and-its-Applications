from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np


class sequential(object):
    def __init__(self, *args):
        """
        Sequential Object to serialize the NN layers
        Please read this code block and understand how it works
        """
        self.params = {}
        self.grads = {}
        self.layers = []
        self.paramName2Indices = {}
        self.layer_names = {}

        # process the parameters layer by layer
        for layer_cnt, layer in enumerate(args):
            for n, v in layer.params.items():
                self.params[n] = v
                self.paramName2Indices[n] = layer_cnt
            for n, v in layer.grads.items():
                self.grads[n] = v
            if layer.name in self.layer_names:
                raise ValueError("Existing name {}!".format(layer.name))
            self.layer_names[layer.name] = True
            self.layers.append(layer)

    def assign(self, name, val):
        # load the given values to the layer by name
        layer_cnt = self.paramName2Indices[name]
        self.layers[layer_cnt].params[name] = val

    def assign_grads(self, name, val):
        # load the given values to the layer by name
        layer_cnt = self.paramName2Indices[name]
        self.layers[layer_cnt].grads[name] = val

    def get_params(self, name):
        # return the parameters by name
        return self.params[name]

    def get_grads(self, name):
        # return the gradients by name
        return self.grads[name]

    def gather_params(self):
        """
        Collect the parameters of every submodules
        """
        for layer in self.layers:
            for n, v in layer.params.items():
                self.params[n] = v

    def gather_grads(self):
        """
        Collect the gradients of every submodules
        """
        for layer in self.layers:
            for n, v in layer.grads.items():
                self.grads[n] = v

    def apply_l1_regularization(self, lam):
        """
        Gather gradients for L1 regularization to every submodule
        """
        for layer in self.layers:
            for n, v in layer.grads.items():
                param = self.params[n]
                grad = (param > 0).astype(np.float32) - (param < 0).astype(np.float32)
                self.grads[n] += lam * grad

    def apply_l2_regularization(self, lam):
        """
        Gather gradients for L2 regularization to every submodule
        """
        for layer in self.layers:
            for n, v in layer.grads.items():
                self.grads[n] += lam * self.params[n]


    def load(self, pretrained):
        """
        Load a pretrained model by names
        """
        for layer in self.layers:
            if not hasattr(layer, "params"):
                continue
            for n, v in layer.params.items():
                if n in pretrained.keys():
                    layer.params[n] = pretrained[n].copy()
                    print ("Loading Params: {} Shape: {}".format(n, layer.params[n].shape))

class ConvLayer2D(object):
    def __init__(self, input_channels, kernel_size, number_filters, 
                stride=1, padding=0, init_scale=.02, name="conv"):
        
        self.name = name
        self.w_name = name + "_w"
        self.b_name = name + "_b"

        self.input_channels = input_channels
        self.kernel_size = kernel_size
        self.number_filters = number_filters
        self.stride = stride
        self.padding = padding

        self.params = {}
        self.grads = {}
        self.params[self.w_name] = init_scale * np.random.randn(kernel_size, kernel_size, 
                                                                input_channels, number_filters)
        self.params[self.b_name] = np.zeros(number_filters)
        self.grads[self.w_name] = None
        self.grads[self.b_name] = None
        self.meta = None
    
    def get_output_size(self, input_size):
        '''
        :param input_size - 4-D shape of input image tensor (batch_size, in_height, in_width, in_channels)
        :output a 4-D shape of the output after passing through this layer (batch_size, out_height, out_width, out_channels)
        '''
        output_shape = [None, None, None, None]
        #############################################################################
        # TODO: Implement the calculation to find the output size given the         #
        # parameters of this convolutional layer.                                   #
        #############################################################################
        pass        
        # input_image_shape = [32,28,28,3], kernel_size = 4, n_filt = 16, stride = 2, padding = 3
        batch_size, H, W, in_channels = input_size
        P,K,S = self.padding, self.kernel_size, self.stride
        
        #Formula : output_height = ((i-k+2p)/s)  + 1
        out_height = (H-K+2*P)//S + 1
        out_width = (W-K+2*P)//S + 1
        
        output_shape = [batch_size, out_height, out_width, self.number_filters]
        #############################################################################
        #                             END OF YOUR CODE                              #
        #############################################################################
        return output_shape

    def forward(self, img):
        output = None
        assert len(img.shape) == 4, "expected batch of images, but received shape {}".format(img.shape)

        output_shape = self.get_output_size(img.shape)
        _ , input_height, input_width, _ = img.shape
        _, output_height, output_width, _ = output_shape

        #############################################################################
        # TODO: Implement the forward pass of a single convolutional layer.         #
        # Store the results in the variable "output" provided above.                #
        #############################################################################
        pass

        #Get weight and bias
        w = self.params[self.w_name]
        b = self.params[self.b_name]
        
        #create padded image
        C, N, K, P, S = self.input_channels, self.number_filters, self.kernel_size, self.padding, self.stride

        img_padded = np.pad(img,((0, 0), (P, P), (P, P), (0, 0)), mode='constant')
        batch_size = img.shape[0]
        conv_out = np.zeros((batch_size, output_height , output_width, N),dtype=img.dtype)

        for batch in range(batch_size):
            for jj in range(0,input_width + (2*P)-K + 1,S):
                for ii in range(0,input_height +(2*P)-K + 1,S):
                    for f in range(N):
                        x = img_padded[batch, ii:ii + K, jj : jj+K, :].reshape(-1)
                        wt = w[:,:,:,f].reshape(-1) # [K*K*C,f]
                        bt = b.reshape(-1)
                        conv_out[batch, ii//S, jj//S, f] = np.dot(wt.T,x)+bt[f]
                        
        output = conv_out
        #############################################################################
        #                             END OF YOUR CODE                              #
        #############################################################################
        self.meta = img
        
        return output


    def backward(self, dprev):
        img = self.meta
        if img is None:
            raise ValueError("No forward function called before for this module!")

        dimg, self.grads[self.w_name], self.grads[self.b_name] = None, None, None
        
        #############################################################################
        # TODO: Implement the backward pass of a single convolutional layer.        #
        # Store the computed gradients wrt weights and biases in self.grads with    #
        # corresponding name.                                                       #
        # Store the output gradients in the variable dimg provided above.           #
        #############################################################################
        pass
    
        #Get weight and bias
        W, b= self.params[self.w_name] , self.params[self.b_name]

        # Retrieve hyperparameters and dimensions 
        C, N, K, P, S = self.input_channels, self.number_filters, self.kernel_size, self.padding, self.stride
        batch_size, input_height, input_width,_ = img.shape

        # Initialize dimg, dW, db 
        dimg = np.zeros(img.shape)                           
        dW = np.zeros(W.shape)
        dB = np.zeros((1, 1, 1, N))

        # Pad img and dimg
        img_padded = np.pad(img,((0, 0), (P, P), (P, P), (0, 0)), mode='constant')
        dimg_padded = np.pad(dimg,((0, 0), (P, P), (P, P), (0, 0)), mode='constant')

        #db : dL/db = dL/dO * dO/db = dprev * 1, dw :  dL/dW = dL/dO*do/dW = X * dprev, dx :  dL/dX = dL/dO*dO/dX = W * dprev
        for i in range(batch_size):                       
            for ii in range(0,input_height+(2*P)-K+1, S):                  
                for jj in range(0,input_width+(2*P)-K+1, S):               
                    for f in range(N):          
                        dimg_padded[i,ii:ii+K, jj:jj+K, :] += W[:,:,:,f] * dprev[i, ii//S, jj//S, f]
                        dW[:,:,:,f] += img_padded[i,ii:ii+K, jj:jj+K, :] * dprev[i, ii//S, jj//S, f]
                        dB[:,:,:,f] += dprev[i, ii//S, jj//S, f]

            
        # Remove Padding
        dimg = dimg_padded[:,P:-P, P:-P, :]

        self.grads[self.w_name], self.grads[self.b_name] = dW,dB
        
        #############################################################################
        #                             END OF YOUR CODE                              #
        #############################################################################

        self.meta = None
        return dimg


class MaxPoolingLayer(object):
    def __init__(self, pool_size, stride, name):
        self.name = name
        self.pool_size = pool_size
        self.stride = stride
        self.params = {}
        self.grads = {}
        self.meta = None

    def forward(self, img):
        output = None
        assert len(img.shape) == 4, "expected batch of images, but received shape {}".format(img.shape)
        
        #############################################################################
        # TODO: Implement the forward pass of a single maxpooling layer.            #
        # Store your results in the variable "output" provided above.               #
        #############################################################################
        pass
        
        #create padded image
        K, S = self.pool_size, self.stride

        #Get input and output shapes, to define pool output
        batch_size , input_height, input_width, N = img.shape
        output_height = (input_height-K)//S + 1
        output_width = (input_width-K)//S + 1

        pooling_out = np.zeros((batch_size, output_height , output_width, N),dtype=img.dtype)

        for batch in range(batch_size):
            for jj in range(0,input_width -K + 1,S):
                for ii in range(0,input_height -K + 1,S):
                    for f in range(N):
                        x = img[batch, ii:ii + K, jj : jj+K, :]
                        pooling_out[batch, ii//S, jj//S, f] = np.max(x)
                      
        output = pooling_out
        #############################################################################
        #                             END OF YOUR CODE                              #
        #############################################################################
        self.meta = img

        return output

    def backward(self, dprev):
        img = self.meta

        dimg = np.zeros_like(img)
        _, h_out, w_out, _ = dprev.shape
        h_pool, w_pool = self.pool_size,self.pool_size

        #############################################################################
        # TODO: Implement the backward pass of a single maxpool layer.              #
        # Store the computed gradients in self.grads with corresponding name.       #
        # Store the output gradients in the variable dimg provided above.           #
        #############################################################################
        pass
    
        def find_mask(x):
            mask = (x== np.max(x))
            return mask
        
        #create padded image
        K, S = self.pool_size, self.stride

        #Get input and output shapes, to define pool output
        batch_size , input_height, input_width, C = img.shape
        pooling_out = np.zeros(img.shape)

        for batch in range(batch_size):
            for jj in range(0,input_width -K + 1,S):
                for ii in range(0,input_height -K + 1,S):
                    for c in range(C):
                        x = img[batch, ii:ii + K, jj : jj+K, c]
                        mask = find_mask(x)
                        pooling_out[batch, ii:ii + K, jj : jj+K, c] = np.multiply(mask, dprev[batch, ii//S, jj//S, c])
        
        dimg = pooling_out
        #############################################################################
        #                             END OF YOUR CODE                              #
        #############################################################################
        return dimg
