from math import sqrt
import numpy as np
from NeuralNetStructure.imagetools import ImageTools

class Layer_Input:
    
    def forward(self, inputs, training):
        self.output = inputs

    def backward(self, dOdA):
        pass

class Layer_Dense:

    def __init__(self, n_inputs, n_neurons, weight_regularizer_L1=0,
        weight_regularizer_L2=0, bias_regularizer_L1=0, 
        bias_regularizer_L2=0):

        # transpose of the weight matrix so it need not be transposed later
        self.weights = 0.1 * np.random.randn(n_inputs, n_neurons)
        self.biases = np.zeros((1, n_neurons))

        self.weight_regularizer_L1 = weight_regularizer_L1
        self.weight_regularizer_L2 = weight_regularizer_L2
        self.bias_regularizer_L1 = bias_regularizer_L1
        self.bias_regularizer_L2 = bias_regularizer_L2

    def forward(self, inputs, training):

        self.inputs = inputs
        self.output = np.dot(inputs, self.weights) + self.biases

    def backward(self, dOdA):
        
        self.dOdW = np.dot(self.inputs.T, dOdA)

        # keeping row vector form
        self.dOdB = np.sum(dOdA, axis=0, keepdims=True)

        # loss adds the L1/L2 norms
        if self.weight_regularizer_L1 > 0:
            dL1 = np.ones_like(self.weights)
            dL1[self.weights < 0] = -1
            self.dOdW += self.weight_regularizer_L1 * dL1

        if self.weight_regularizer_L2 > 0:
            self.dOdW += 2 * self.weight_regularizer_L2 * self.weights

        if self.bias_regularizer_L1 > 0:
            dL1 = np.ones_like(self.biases)
            dL1[self.biases < 0] = -1
            self.dOdB += self.bias_regularizer_L1 * dL1

        if self.bias_regularizer_L2 > 0:
            self.dOdB += 2 * self.bias_regularizer_L2 * self.biases

        # to be passed back as dOdA in previous layers backwards method
        self.dOdI = np.dot(dOdA, self.weights.T)

    def get_parameters(self):
        return self.weights, self.biases
    
    def set_parameters(self, weights, biases):
        self.weights = weights
        self.biases = biases

class Conv_Input:
    
    pass

class Layer_Conv:

    def __init__(self, input_channels,  num_filters, filter_dims=(3, 3), stride=(1, 1), dilation=(1, 1), 
        convolve_type="VALID",
        weight_regularizer_L1=0, weight_regularizer_L2=0, bias_regularizer_L1=0, bias_regularizer_L2=0):
        
        self.input_channels = input_channels
        self.num_filters = num_filters
        self.filter_w = filter_dims[1]
        self.filter_h = filter_dims[0]
        self.stride = stride
        self.dilation = dilation
        self.convolve_type = convolve_type
        
        SD = input_channels * self.filter_h * self.filter_w
        SD = sqrt(2 / SD)
        
        # This will be updated after back propogation. Weights are filters. 
        self.weights = np.random.randint(-1, 2, (self.num_filters,
                                                self.input_channels, 
                                                self.filter_h, 
                                                self.filter_w)).astype(np.float64)
        self.biases = np.zeros((1, self.num_filters, 1, 1)).astype(np.float64)

        self.weight_regularizer_L1 = weight_regularizer_L1
        self.weight_regularizer_L2 = weight_regularizer_L2
        self.bias_regularizer_L1 = bias_regularizer_L1
        self.bias_regularizer_L2 = bias_regularizer_L2
    
    def update_dims(self, dims):
        if self.convolve_type == "VALID":
            next_dims_w = int(((dims[2] - self.filter_w) / self.stride[0]) + 1)
            next_dims_h = int(((dims[1] - self.filter_h) / self.stride[1]) + 1)
            return self.num_filters, next_dims_h, next_dims_w
        else:
            return dims

    def forward(self, images, training):

        # image needs to be a numpy array, with channel first format, and with number of imput channels
        # this layer specifies during construction
        # outputs are the feature maps
        
        self.inputs = images

        self.output = np.stack(
                                [
                                    ImageTools.IMGconvolveMultiKernel(
                                                            image, 
                                                            kernels=self.weights,
                                                            stride=self.stride,
                                                            dilation=self.dilation,
                                                            convolvetype=self.convolve_type
                                                            )
                                    for image in images
                                ]
                                )
        
        # adding biases per filter
        keys = np.arange(self.num_filters)
        self.output[:, keys, :, :] += self.biases[:, keys, :, :]

    def backward(self, dOdA):

  
        D = self.weights.shape[0] # num filters, also num features in dOdA
        L = self.weights.shape[1] # num channels per filter/image
        K_h = self.weights.shape[2]
        K_w = self.weights.shape[3]

        D_h, D_w = self.dilation[0], self.dilation[1]
        S_h, S_w = self.stride[0], self.stride[1]

        dOdA_ = ImageTools.dilate(dOdA, (S_h, S_w))
        self.weights_ = ImageTools.dilate(self.weights, (D_h, D_w))

        self.dOdW = np.stack(
            [
                np.stack(
                    [
                        ImageTools.IMGconvolve(
                            self.inputs[:, image_channel, : :],
                            dOdA_[:, dOdA__channel, : :],
                            stride=(D_h, D_w), 
                            dilation=(1, 1), 
                            convolvetype="VALID")
                        for image_channel in range(L)
                    ]
                )
                for dOdA__channel in range(D)
            ]

        )

        flippedfilters_ = np.rot90(self.weights_, 2, (2, 3))
        flippedfilters = np.rot90(self.weights, 2, (2, 3)) 

        self.dOdB = np.array([np.stack(
            [
                [[np.sum(dOdA[:, z, :, :])]] for z in range(self.num_filters)
            ]
        )])

        # loss adds the L1/L2 norms
        if self.weight_regularizer_L1 > 0:
            dL1 = np.ones_like(self.weights)
            dL1[self.weights < 0] = -1
            self.dOdW += self.weight_regularizer_L1 * dL1

        if self.weight_regularizer_L2 > 0:
            self.dOdW += 2 * self.weight_regularizer_L2 * self.weights

        if self.bias_regularizer_L1 > 0:
            dL1 = np.ones_like(self.biases)
            dL1[self.biases < 0] = -1
            self.dOdB += self.bias_regularizer_L1 * dL1

        if self.bias_regularizer_L2 > 0:
            self.dOdB += 2 * self.bias_regularizer_L2 * self.biases


        self.dOdI = np.stack([
                        np.stack(
                                    [
                                        ImageTools.IMGconvolve
                                        (
                                        dOdA_[img_n, :, :, :], 
                                        flippedfilters_[:, img_channel , :, :],
                                        stride=(1, 1),
                                        dilation=(1, 1), 
                                        convolvetype="FULL"
                                        )
                                    for img_channel in range(self.inputs.shape[1])
                                    ]
                                )
                            for img_n in range(self.inputs.shape[0])
                        ]
                    )

    def get_parameters(self):
        return self.weights, self.biases
    
    def set_parameters(self, weights, biases):
        self.weights = weights
        self.biases = biases
    
class Layer_Flatten:
    '''
    Not trainable. 
    '''
    def __init__(self):
        pass

    def forward(self, input, training):
        self.inputs = input
        self.output = np.array([input[i].flatten() for i in range(len(input))])
    
    def backward(self, dOdA):
        self.dOdI = np.reshape(dOdA, self.inputs.shape) 