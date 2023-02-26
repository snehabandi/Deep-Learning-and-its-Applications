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
                        x = img_padded[batch, ii:ii + K, jj : jj+K, :].reshape(-1,1)
                        wt = w[:,:,:,f].reshape(-1,1) # [K*K*C,f]
                        conv_out[batch, ii//S, jj//S, f] = np.dot(wt.T,x)+b[f]
                        
        output = conv_out
        #optimized code
        # for jj in range(0,input_width -K + 1,S):
        #     for ii in range(0,input_height -K + 1,S):
        #         for f in range(N):
        #             x = img_padded[:, ii:ii + K, jj : jj+K, :] .reshape(batch_size,-1)#[B,H*W*C]
        #             wt = w[:,:,:,f].reshape(-1) # [K*K*C,N]
        #             conv_out[:, ii//S, jj//S, f] = np.dot(x,wt)+b[f]
                        
        # output = conv_out
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
        ### START CODE HERE ###

        # Retrieve information from "cache"
        A_prev, W, b, dZ = img, self.params[self.w_name] , self.params[self.b_name], dprev

        # Retrieve dimensions from A_prev's shape
        (m, n_H_prev, n_W_prev, n_C_prev) = A_prev.shape

        # Retrieve dimensions from W's shape
        (f, f, n_C_prev, n_C) = W.shape

        # Retrieve information from "hparameters"
        stride = self.stride
        pad = self.padding

        # Retrieve dimensions from dZ's shape
        (m, n_H, n_W, n_C) = dZ.shape

        # Initialize dA_prev, dW, db with the correct shapes
        dA_prev = np.zeros((m, n_H_prev, n_W_prev, n_C_prev))                           
        dW = np.zeros((f, f, n_C_prev, n_C))
        dB = np.zeros((1, 1, 1, n_C))
        dimg = np.zeros((m, n_H_prev, n_W_prev, n_C_prev))                           

        # Pad A_prev and dA_prev
        A_prev_pad = np.pad(A_prev,((0, 0), (pad, pad), (pad, pad), (0, 0)), mode='constant')
        dA_prev_pad = np.pad(dA_prev,((0, 0), (pad, pad), (pad, pad), (0, 0)), mode='constant')

        for i in range(m):                       # loop over the training examples

            # select ith training example from A_prev_pad and dA_prev_pad
            a_prev_pad = A_prev_pad[i]
            da_prev_pad = dA_prev_pad[i]

            for h in range(n_H):                   # loop over vertical axis of the output volume
                for w in range(n_W):               # loop over horizontal axis of the output volume
                    for c in range(n_C):           # loop over the channels of the output volume

                        # Find the corners of the current "slice"
                        vert_start  = h*stride
                        vert_end    = vert_start + f
                        horiz_start = w*stride
                        horiz_end   = horiz_start + f

                        # Use the corners to define the slice from a_prev_pad
                        a_slice = a_prev_pad[vert_start:vert_end, horiz_start:horiz_end, :]

                        # Update gradients for the window and the filter's parameters using the code formulas given above
                        da_prev_pad[vert_start:vert_end, horiz_start:horiz_end, :] += W[:,:,:,c] * dZ[i, h, w, c]
                        dW[:,:,:,c] += a_slice * dZ[i, h, w, c]
                        dB[:,:,:,c] += dZ[i, h, w, c]
#                         w_re = w[:,:, :, f].reshape(-1,w.shape[2])               #(4,4,3,12)
#                         dL_dX = dprev[i,:,:,c].reshape(-1,1)         # (15, 4, 4, 12)
#                         da_prev_pad[i, h, w, :] += np.dot(dL_dX.T,w_re) # (15,8,8,3)
            
            # Set the ith training example's dA_prev to the unpaded da_prev_pad (Hint: use X[pad:-pad, pad:-pad, :])
            if pad!=0:
                dimg[i] = da_prev_pad[pad:-pad, pad:-pad, :]
            else:
                dimg[i] = da_prev_pad
        ### END CODE HERE ###
            # dB = np.sum(dZ, axis=(0, 1,2)) 
    
            dimg, self.grads[self.w_name], self.grads[self.b_name] = dimg,dW,dB

    
    
#         batch_size, input_height, input_width, _ = img.shape
        
#         C, N, K, P, S = self.input_channels, self.number_filters, self.kernel_size, self.padding, self.stride

#         #Get weight and bias
#         w = self.params[self.w_name]
# #         b = self.params[self.b_name]

#         #db : dL/db = dL/dO * dO/db = dprev * 1
#         db = np.sum(dprev, axis=(0, 1,2))
        
#         #dw :  dL/dW = dL/dO*do/dW = X * dprev
#         img_padded = np.pad(img,((0, 0), (P, P), (P, P), (0, 0)), mode='constant')                                    
#         dw = np.zeros((K ,K ,C, N),dtype=img.dtype)

#         for batch in range(batch_size):
#             for f in range(N):
#                 dL_dW = dprev[batch,:,:,f].reshape(-1,1) 
#                 for ii in range(0,input_height + (2*P)-K+1,S):
#                     for jj in range(0,input_width+ (2*P)-K+1,S):
#                         for c in range(C):
#                             x = img_padded[batch, ii:ii + K, jj : jj+K, c].reshape(-1,1)
#                             dw[ii//S, jj//S,c, f] += np.dot(dL_dW.T,x)
                            
#         #dx :  dL/dX = dL/dO*dO/dX = W * dprev
#         dx = np.zeros((img.shape),dtype=img.dtype)
#         dx_padded = np.zeros((15,10,10,3),dtype=img.dtype)
        
#         print(dx_padded.shape)
#         for batch in range(batch_size):
#             for ii in range(0,input_height + (2*P)-K+1,S):
#                 for jj in range(0,input_width+ (2*P)-K+1,S):
#                     for c in range(C):
#                         for f in range(N):
#                             dL_dX = dprev[batch,:,:,f].reshape(-1,1) 
#                             w_re = w[:,:, c, f].reshape(-1,1)
#                             dx_padded[batch, ii//S, jj//S, c] = np.dot(dL_dX.T,w_re)

#         dx_depadded = dx_padded[:, P:-P,P:-P, :]

#         dimg, self.grads[self.w_name], self.grads[self.b_name] = dx_depadded,dw,db
        
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
        A_prev = img
        
        # Retrieve dimensions from the input shape
        (m, n_H_prev, n_W_prev, n_C_prev) = A_prev.shape

        # Retrieve hyperparameters from "hparameters"
        f = self.pool_size
        stride = self.stride

        # Define the dimensions of the output
        n_H = int(1 + (n_H_prev - f) / stride)
        n_W = int(1 + (n_W_prev - f) / stride)
        n_C = n_C_prev

        # Initialize output matrix A
        A = np.zeros((m, n_H, n_W, n_C)) 
        for i in range(m):                         # loop over the training examples
            for h in range(n_H):                     # loop on the vertical axis of the output volume
                for w in range(n_W):                 # loop on the horizontal axis of the output volume
                    for c in range (n_C):            # loop over the channels of the output volume

                        # Find the corners of the current "slice" (≈4 lines)
                        vert_start = h * stride
                        vert_end   = vert_start + f
                        horiz_start = w * stride
                        horiz_end   = horiz_start + f

                        # Use the corners to define the current slice on the ith training example of A_prev, channel c. (≈1 line)
                        a_prev_slice = A_prev[i, vert_start:vert_end, horiz_start:horiz_end, c]

                        # Compute the pooling operation on the slice. Use an if statment to differentiate the modes. Use np.max/np.mean.
                        A[i, h, w, c] = np.max(a_prev_slice)
                        
        output = A
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
    
        def create_mask_from_window(x):
            mask = (x== np.max(x))
            return mask
        
        ### START CODE HERE ###
    
        # Retrieve information from cache (≈1 line)
        A_prev, dA = img, dprev

        # Retrieve hyperparameters from "hparameters" (≈2 lines)
        stride = self.stride
        f = self.pool_size

        # Retrieve dimensions from A_prev's shape and dA's shape (≈2 lines)
        m, n_H_prev, n_W_prev, n_C_prev = A_prev.shape
        m, n_H, n_W, n_C = dA.shape

        # Initialize dA_prev with zeros (≈1 line)
        dA_prev = np.zeros(A_prev.shape)

        for i in range(m):                       # loop over the training examples
            # select training example from A_prev (≈1 line)
            a_prev = A_prev[i]
            for h in range(n_H):                   # loop on the vertical axis
                for w in range(n_W):               # loop on the horizontal axis
                    for c in range(n_C):           # loop over the channels (depth)
                        # Find the corners of the current "slice" (≈4 lines)
                        vert_start  = h*stride
                        vert_end    = vert_start + f
                        horiz_start = w*stride
                        horiz_end   = horiz_start + f

                        # Use the corners and "c" to define the current slice from a_prev (≈1 line)
                        a_prev_slice = a_prev[vert_start:vert_end, horiz_start:horiz_end, c]
                        # Create the mask from a_prev_slice (≈1 line)
                        mask = create_mask_from_window(a_prev_slice)
                        # Set dA_prev to be dA_prev + (the mask multiplied by the correct entry of dA) (≈1 line)
                        dA_prev[i, vert_start:vert_end, horiz_start:horiz_end, c] += np.multiply(mask, dA[i, h, w, c])
        
        dimg = dA_prev
        #############################################################################
        #                             END OF YOUR CODE                              #
        #############################################################################
        return dimg
