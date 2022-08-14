import numpy as np
from NeuralNetStructure.imagetools import ImageTools
class Pool:

    def __init__(self, pool_dims=(1, 1), stride=(1, 1), pool_type='MAX'):
        
        self.pool_dims = pool_dims
        self.stride = stride
        self.pool_type = pool_type

    def update_dims(self, dims):
        next_dims_w = int(((dims[2] - self.pool_dims[0]) / self.stride[0]) + 1)
        next_dims_h = int(((dims[1]- self.pool_dims[1]) / self.stride[1]) + 1)
        return dims[0], next_dims_h, next_dims_w

    def forward(self, images, training):
        self.input_shape = images.shape
        self.mask = np.array(images.shape)

        pool_return = [ImageTools.pool(
                                            image, 
                                            kernel_dims=self.pool_dims, 
                                            stride=self.stride,
                                            pool_type=self.pool_type) 
                                    for image in images]

        feature_maps = tuple(pool_return[index][0] for index in range(len(images)))
        self.max_masks = [pool_return[index][1] for index in range(len(images))] # a list of dicts per image

        self.output = np.stack(feature_maps)

    def backward(self, dvalues):
        
        self.dOdI = np.zeros(self.input_shape)
        for image_index, image in enumerate(dvalues):
                image_masks = self.max_masks[image_index]
                for max_pixel_num, max_pixel in enumerate(image_masks):
                    for contribution in image_masks[max_pixel]:
                        self.dOdI[(image_index,) + max_pixel] += image[(max_pixel[0],) + contribution]
        return self.dOdI