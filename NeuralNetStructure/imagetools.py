r"""
# Notice: This module adopts a rigid philosophy to make handling of images consistent.
# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#           1. To methods, ALL images will be 3D numpy arrays, REGARDLESS of the number of channels. If a 
#              method receives a 2D array, convert it to 3D. This generalizes the method.  
#           2. Images are written with CHANNEL FIRST convention, then height, then width. 
#           3. Methods will NOT accept diverse inputs, specify inputs before hand instead of
#              forcing methods to handle them. This eliminates ambiguity of the purpose of the method. 
#           4. Methods ALWAYS return 3D numpy arrays, if they dont they are a HELPER.  
"""
import numpy as np



class ConvolveException(Exception):
        pass

class PoolException(Exception):
        pass

class ImageTools:

    # KERNELS ==========================================
    identity = np.array([[0, 0, 0], [0, 1, 0], [0, 0, 0]])
    bottomSobel = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])
    topSobel = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])
    leftSobel = np.array([[1, 0, -1], [2, 0, -2], [1, 0, -1]])
    rightSobel = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    outline = np.array([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]])
    sharpen = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
    emboss = np.array([[0,-1,-1], [1,0,-1], [1,1,0]])
    # ============================================================

    @staticmethod
    def convolveCheck(image, kernel):

        if len(kernel.shape) != len(image.shape):
            raise ConvolveException(f'Kernel shape ({kernel.shape}) doesnt match image shape ({image.shape})')
            
        if len(kernel.shape) == 3 and len(image.shape) == 3:
            if kernel.shape[0] != image.shape[0]:
                raise ConvolveException(f'Kernel channels ({kernel.shape[0]}) do not match image channels ({image.shape[0]})')
            
        if kernel.shape[-1] > image.shape[-1] or kernel.shape[-2] > image.shape[-2] :
            raise ConvolveException(f'Kernel (shape = {kernel.shape}) is too big (image shape = {image.shape})')
    
    @staticmethod
    def poolCheck(image, kernel_dims):

        if kernel_dims[-1] > image.shape[-1] or kernel_dims[-2] > image.shape[-2] :
            raise PoolException(f'Kernel (shape = {kernel_dims}) is too big (image shape = {image.shape})')
    
    
    
    @staticmethod
    def channelLastToFirst(img):
        img = np.moveaxis(img, 2, 0)
        return img
    @staticmethod
    def channelFirstToLast(img):
        img = np.moveaxis(img, 0, 2)
        return img

    @staticmethod
    def normalize(img, type="MinMax"):

        if len(img.shape) == 2:
            img = np.expand_dims(img, 0)

        data_min = np.min(img, axis=(1, 2), keepdims=True)
        data_max = np.max(img, axis=(1, 2), keepdims=True)
        data_mean = np.mean(img, axis=(1, 2), keepdims=True)
        stdev = np.std(img, axis=(1, 2), keepdims=True)
        
        if type=="Mean":
            img = (img - data_mean) / stdev
        else:
            img = (img - data_min) / (data_max - data_min)
        
        return img

        

    @staticmethod
    def ArrayConvolve(array, kernel, stride=(1, 1), dilation=(1, 1), convolve_type="VALID"):
        
        I_h, I_w    =   array.shape[0],  array.shape[1]
        S_h, S_w    =   stride[0],       stride[1]
        K_h, K_w    =   kernel.shape[0], kernel.shape[1]
        D_h, D_w    =   dilation[0],     dilation[1]

        if dilation != (1, 1):
            kernel__  = ImageTools.dilate(kernel, (D_h, D_w))
        else:
            kernel__ = kernel
        K__h, K__w  = D_h * (K_h - 1) + 1, D_w * (K_w - 1) + 1
        

        # ZERO PADDING =================================================
        p_h, p_w    = 0, 0

        if convolve_type == "VALID":
            pass

        elif convolve_type == "SAME": 
            # Ω = (I_h - K__h + 2p) / S_h + 1 = I_h
            # <=> S_h * I_h = S_h + I_h - K__h  + 2p
            # <=> p = (S_h * I_h - S_h - I_h + K__h) / 2
            p_h, p_w = (S_h * I_h - S_h - I_h + K__h), (S_w * I_w - S_w - I_w + K__w)

            if p_h % 2 and p_w % 2:
                print(p_h)
                raise PoolException("Cannot \"same\" pad with given dimensions")
            else:
                p_h, p_w = int(p_h / 2), int(p_w / 2)

        elif convolve_type == "FULL":
            p_h, p_w = K__h - 1, K__w - 1

        else:
            raise ConvolveException(f'Invalid convolution type ({convolve_type})')

        array = np.pad(array, [(p_h, p_h), (p_w, p_w)])
        # =================================================================

        # Convolution 
        output = []

        for y in range(0, I_h + (2 * p_h) - K_h + 1, S_h): #
            row = []
            
            for x in range(0, I_w + (2 * p_w) - K__w + 1, S_w): # 
                
                receptive_field = array[y: y + K__h, x: x + K__w]

                element_wise_product = receptive_field * kernel__
                pixel = np.concatenate(element_wise_product).sum()
                
                row.append(pixel)
            
            output.append(row) 
        return np.array(output) 


    @staticmethod
    def IMGconvolve(image, kernel, stride=(1, 1), dilation=(1,1), convolvetype="VALID"):
        '''
        Operates on a single kernel. 
        Image and kernel are 3D numpy arrays.
        Will naturally return a 2D numpy array, since it is a spatial convolution. 
        '''
        # make image and kernel 3D if they are 2D, since this also handles multi-channel input
        if len(image.shape) == 2:
            image = np.expand_dims(image, 0)
        if len(kernel.shape) == 2:
            kernel = np.stack([kernel for i in range(3)])

        # check dimensions
        ImageTools.convolveCheck(image, kernel)
        
        return np.sum(
                        tuple(ImageTools.ArrayConvolve(
                                    image[i], 
                                    kernel[i], 
                                    stride=stride, 
                                    dilation=dilation, 
                                    convolve_type=convolvetype) 
                            for i in range(image.shape[0])), 
                    0)
       
    
    @staticmethod
    def IMGconvolveMultiKernel(image, kernels, stride=(1, 1), dilation=(1, 1), convolvetype="VALID"):
        '''
        Operates on multiple kernels and will return a 3D numpy array. 
        '''
        return np.stack(
                            tuple(ImageTools.IMGconvolve(
                                                                image=image, 
                                                                kernel=kernel, 
                                                                stride=stride,
                                                                dilation=dilation,
                                                                convolvetype=convolvetype
                                                                ) 
                            for kernel in kernels)
                            )

    @staticmethod 
    def pool(image, kernel_dims=(2, 2), stride=(1, 1), pool_type='MAX'):

        '''
        Image will be forced into a 3D numpy array to account for channels. 
        Pool kernel however is a 2D numpy array. This is because pooling applies to each image channel
        separately. Due to channel independent pooling, this returns a 3D numpy array. 
        If a greyscale is returned, reduce dims by 1 to properly format the image. 
        '''
        if len(image.shape) == 2:
            image = np.expand_dims(image, 0)
        # check dimensions
        ImageTools.poolCheck(image, kernel_dims)
        
        
        # set dimentions variables
        I_h, K_h = image.shape[-2], kernel_dims[0]
        I_w, K_w = image.shape[-1], kernel_dims[1]
        S_h, S_w = stride[0], stride[1]

        # set pool type
        pool = np.max if pool_type == 'MAX' else np.mean
        pool_output = []
        max_mask = {} # dict of which pixels get chosen (keys) matching how many times they get chosen, 
        # and what index the result is in in the feature map
        for channel in range(len(image)):
            channel_output = []

            for y in range(0, I_h - K_h + 1, S_h): 
                feature_map_row = []
                
                for x in range(0, I_w - K_w + 1, S_w): 
                
                    receptive_field = image[channel][
                        y: y + K_h, 
                        x: x + K_w]
                    
                    pool_pixel, pool_pixel_index = pool(receptive_field), np.unravel_index(np.argmax(image[channel, 
                                                                y: y + K_h, 
                                                                x: x + K_w], 
                                                                axis=None), (K_h, K_w))
                    image_pixel_index = (channel, pool_pixel_index[0] + y, pool_pixel_index[1] + x)                                          
                    if image_pixel_index not in max_mask:
                        max_mask[image_pixel_index] = [(len(channel_output), len(feature_map_row))]
                    else:
                        max_mask[image_pixel_index].append((len(channel_output), len(feature_map_row)))
                    
                    feature_map_row.append(pool_pixel)
                channel_output.append(feature_map_row)
            pool_output.append(channel_output)
        return np.array(pool_output), max_mask

    def pool_basic(image, kernel_dims, stride, pool_type='MEAN'):

        '''
        Will not keep track of kept pixels in the case of MAX pooling.  
        '''
        if len(image.shape) == 2:
            image = np.expand_dims(image, 0)
        # check dimensions
        ImageTools.poolCheck(image, kernel_dims)
        
        
        # set dimentions variables
        I_h, K_h = image.shape[-2], kernel_dims[0]
        I_w, K_w = image.shape[-1], kernel_dims[1]
        S_h, S_w = stride[0], stride[1]

        # set pool type
        pool = np.max if pool_type == 'MAX' else np.mean
        pool_output = []
    
        for channel in range(len(image)):
            channel_output = []

            for y in range(0, I_h - K_h + 1, S_h): 
                feature_map_row = []
                
                for x in range(0, I_w - K_w + 1, S_w): 
                
                    receptive_field = image[channel][
                        y: y + K_h, 
                        x: x + K_w]
                    
                    pool_pixel = pool(receptive_field) 
                    feature_map_row.append(pool_pixel)
                channel_output.append(feature_map_row)
            pool_output.append(channel_output)
        return np.array(pool_output)

    def dilate(img, dilation):
        spatial_dims = img.shape[-2:]
        I_h, I_w = spatial_dims[0], spatial_dims[1]
        D_h, D_w = dilation[0], dilation[1]

        I__h, I__w = D_h * (I_h - 1) + 1, D_w * (I_w - 1) + 1
        img_shape = img.shape
        dilated_shape = img_shape[:-2] + (I__h, I__w)
        dilated = np.zeros(dilated_shape)

        indeces = np.ndindex(img_shape)
        for index in indeces:
            index_ = index[:-2] + (int(index[-2] * D_h), int(index[-1] * D_w))
            dilated[index_] = img[index]
        return dilated
            
            
            
            

            
                




    def bilinear_interpolate(img, target_dims):

        """img contains 3 channels"""
        
        virtual_step_w = (target_dims[1] - 1) / (img.shape[2] - 1)
        virtual_step_h = (target_dims[0] - 1) / (img.shape[1] - 1)


        new_img = []
        for n_channel in range(3):
            channel = np.zeros(target_dims)

            t_curr_pixel_h = 0

            t_curr_checkpoint_h = 0
            t_next_checkpoint_h = virtual_step_h

            i_curr_checkpoint_h = 0
            i_next_checkpoint_h = 1

            while t_curr_pixel_h < target_dims[0]:

                while t_curr_pixel_h < t_next_checkpoint_h and t_curr_pixel_h < target_dims[0]:

                    t_curr_pixel_w = 0
                    t_curr_checkpoint_w = 0
                    t_next_checkpoint_w = virtual_step_w

                    i_curr_checkpoint_w = 0
                    i_next_checkpoint_w = 1
                    while t_curr_pixel_w < target_dims[1]:

                        while t_curr_pixel_w < t_next_checkpoint_w and t_curr_pixel_w < target_dims[1]:
                           
                            f11 = img[n_channel, i_curr_checkpoint_h, i_curr_checkpoint_w]
    
                            mat_x = np.array([[t_next_checkpoint_w - t_curr_pixel_w, t_curr_pixel_w - t_curr_checkpoint_w]])
                            mat_y = np.array([[t_next_checkpoint_h - t_curr_pixel_h], [t_curr_pixel_h - t_curr_checkpoint_h]])

                            coeff_x = 1 / (t_next_checkpoint_w - t_curr_checkpoint_w)
                            coeff_y = 1 / (t_next_checkpoint_h - t_curr_checkpoint_h)

                            if i_curr_checkpoint_h < img.shape[1] - 1 and i_curr_checkpoint_w < img.shape[2] - 1:
                                
                                f12 = img[n_channel, i_next_checkpoint_h, i_curr_checkpoint_w]
                                f21 = img[n_channel, i_curr_checkpoint_h, i_next_checkpoint_w]
                                f22 = img[n_channel, i_next_checkpoint_h, i_next_checkpoint_w]

                                mat_f = np.array([[f11, f12], [f21, f22]])
                                mat = np.dot(mat_f, mat_y)
                                mat = np.dot(mat_x, mat)

                                channel[t_curr_pixel_h, t_curr_pixel_w] = coeff_x * coeff_y * mat
                            
                            elif i_curr_checkpoint_w == img.shape[2] - 1 and i_curr_checkpoint_h < img.shape[1] - 1:
                                
                                f12 = img[n_channel, i_next_checkpoint_h, i_curr_checkpoint_w]
                                mat_f = np.array([[f11, f12]])
                                channel[t_curr_pixel_h, t_curr_pixel_w] = coeff_y * np.dot(mat_f, mat_y)

                            elif i_curr_checkpoint_h == img.shape[1] - 1 and i_curr_checkpoint_w < img.shape[2] - 1:

                                f21 = img[n_channel, i_curr_checkpoint_h, i_next_checkpoint_w]
                                mat_f = np.array([[f11], [f21]])
                                channel[t_curr_pixel_h, t_curr_pixel_w] = coeff_x * np.dot(mat_x, mat_f)

                            else:
                                channel[t_curr_pixel_h, t_curr_pixel_w] = img[n_channel, i_curr_checkpoint_h, i_curr_checkpoint_w]


                            t_curr_pixel_w += 1

                        t_curr_checkpoint_w += virtual_step_w
                        t_next_checkpoint_w += virtual_step_w
                        i_curr_checkpoint_w += 1
                        i_next_checkpoint_w += 1
              
                    t_curr_pixel_h += 1

                t_curr_checkpoint_h += virtual_step_h
                t_next_checkpoint_h += virtual_step_h
                i_curr_checkpoint_h += 1
                i_next_checkpoint_h += 1

            new_img.append(channel)
        
        return np.array(new_img)

    @staticmethod
    def scale_down(img, target_dims):
        slide_w = int(img.shape[1] / target_dims[0])
        slide_h = int(img.shape[2] / target_dims[1])
        return ImageTools.pool_basic(img, (slide_w, slide_h), (slide_w, slide_h), "MEAN")
        

    @staticmethod
    def scale_array(img, target_dims):
        img_dims = img.shape

        if img_dims[0] == target_dims[0]:
            return img
        elif img_dims[0] < target_dims[0]:
            return ImageTools.bilinear_interpolate(img, target_dims)
        else:
            return ImageTools.scale_down(img, target_dims)
