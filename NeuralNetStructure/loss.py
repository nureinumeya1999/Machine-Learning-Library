import numpy as np

class Loss:

    def remember_trainable_layers(self, trainable_layers):
        
        self.trainable_layers = trainable_layers

    def regularization_loss(self):

        reg_loss = 0

        for layer in self.trainable_layers:
            if layer.weight_regularizer_L1 > 0:
                reg_loss += layer.weight_regularizer_L1 * np.sum(np.abs(layer.weights))

            if layer.weight_regularizer_L2 > 0:
                reg_loss += layer.weight_regularizer_L2 * np.sum(layer.weights ** 2)
            
            if layer.bias_regularizer_L1 > 0:
                reg_loss += layer.bias_regularizer_L1 * np.sum(np.abs(layer.biases))

            if layer.bias_regularizer_L2 > 0:
                reg_loss += layer.bias_regularizer_L2 * np.sum(layer.biases ** 2)
            
        return reg_loss 

    def new_pass(self):
        self.accumulated_sum = 0
        self.accumulated_count = 0

    def calculate(self, output, y, *, include_regularization=False):

        sample_losses = self.forward(output, y)
        data_loss = np.mean(sample_losses)
        reg_loss = self.regularization_loss()

        self.accumulated_sum += np.sum(sample_losses)
        self.accumulated_count += len(sample_losses)
        
        if not include_regularization:
            return data_loss
        return data_loss, reg_loss

    def calculate_accumulated(self, *, include_regularization=False):

        data_loss = self.accumulated_sum / self.accumulated_count

        if not include_regularization:
            return data_loss
        
        return data_loss, self.regularization_loss()
    

class Loss_MeanSquaredError(Loss):

    #for regression

    def forward(self, y_pred, y_true):

        sample_losses = np.mean((y_true - y_pred)**2, axis=1)
        return sample_losses

    def backward(self, dvalues, y_true):

        samples = len(dvalues)
        outputs = len(dvalues[0])
        self.dOdI = -2 * (y_true - dvalues) / outputs
        self.dOdI /= samples

class Loss_MeanAbsoluteError(Loss):

    # for regression

    def forward(self, y_pred, y_true):

        sample_losses  = np.mean(np.abs(y_true - y_pred), axis=1)
        return sample_losses
    
    def backward(self, dvalues, y_true):

        samples = len(dvalues)
        outputs = len(dvalues[0])

        self.dOdI = np.sign(y_true - dvalues) / outputs
        self.dOdI /= samples

        
class Loss_BinaryCrossentropy(Loss):

    # for classification

    def forward(self, y_pred, y_true):

        y_pred_clipped = np.clip(y_pred, 1e-7, 1-1e-7)
        sample_losses = -(y_true * np.log(y_pred_clipped) + (1-y_true) * np.log(1 - y_pred_clipped))
        sample_losses = np.mean(sample_losses, axis=1)
        return sample_losses

    def backward(self, dvalues, y_true):

        samples = len(dvalues)
        outputs = len(dvalues[0])

        clipped_dvalues = np.clip(dvalues, 1e-7, 1-1e-7)
        self.dOdI = -(y_true / clipped_dvalues - (1 - y_true) / (1 - clipped_dvalues)) / outputs
        self.dOdI = self.dOdI / samples 

class Loss_CategoricalCrossentropy(Loss):

    # for classification

    def forward(self, y_pred, y_true):

        samples = len(y_pred)

        # clip to avoid log(0) error
        y_pred_clipped = np.clip(y_pred, 1e-7, 1 - 1e-7)

        # return list of each samples activated neuron score 
        if len(y_true.shape) == 1: 
            correct_confidences = y_pred_clipped[range(samples), y_true]
        
        else: # one hot encoded ground truths
            correct_confidences = np.sum(y_pred_clipped * y_true, axis=1)

        # calculate log loss, the final metric used to train the model, per sample. That is, a (batch_size, 1) dimensional vector. 
        negative_log_likelihoods = - np.log(correct_confidences)
        return negative_log_likelihoods
    
    def backward(self, dvalues, y_true):

        num_samples = len(dvalues)
        num_labels = len(dvalues[0])

        # convert to one-hot array
        if len(y_true.shape) == 1: 
            y_true = np.eye(num_labels)[y_true]
        
        # gradient 
        self.dOdI = -y_true / dvalues

        # normalize
        self.dOdI = self.dOdI / num_samples

    