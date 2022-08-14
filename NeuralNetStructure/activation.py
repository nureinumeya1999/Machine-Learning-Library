import numpy as np
from NeuralNetStructure.loss import Loss_CategoricalCrossentropy

class Activation_Linear:
    # for regression models

    def forward(self, inputs, training):
        self.inputs = inputs
        self.output = inputs
    
    def backward(self, dOdA):
        self.dOdI = dOdA.copy()

    def predictions(self, outputs):
        return outputs 



class Activation_ReLU:

    # for hidden layers

    def forward(self, inputs, training):

        self.inputs = inputs
        self.output = np.maximum(0, inputs)

    def backward(self, dOdA):

        self.dOdI = dOdA.copy()
        
        # gradient of dOdA wrt input
        self.dOdI[self.inputs <= 0] = 0
    
    def predictions(self, outputs):
        return outputs

class Activation_Softmax:

    # for classification

    def forward(self, inputs, training):

        self.inputs = inputs

        # making all values negative, so exponentiation targets (0, 1]
        exp_values = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))

        # softmax calculation 
        probabilities = exp_values / np.sum(exp_values, axis=1, keepdims=True)
        self.output = probabilities

    def backward(self, dOdA):

        # same shape as dOdA 
        self.dOdI = np.empty_like(dOdA)

        # index = row wise calculation 
        for index, (single_output, single_dOdA) in \
            enumerate(zip(self.output, dOdA)):

            # column vector transform of [S_{i, j}]_{j}
            single_output = single_output.reshape(-1, 1)

            # jacobian of softmax wrt [ z_{i, k} ]_{k}  = [ [ delta_{j, k} S_{i, j} - S_{i, j}S_{i, k} ]_{j} ]_{k}
            jacobian_matrix = np.diagflat(single_output) - \
                 np.dot(single_output, single_output.T)
            self.dOdI[index] = np.dot(jacobian_matrix, single_dOdA)

    def predictions(self, outputs):
        return np.argmax(outputs, axis=1, keepdims=False)

class Activation_Sigmoid:

    # for binary classification

    def forward(self, inputs, training):

        self.inputs = inputs
        self.output = 1/(1 + np.exp(-inputs))
    
    def backward(self, dOdA):

        self.dOdI = dOdA * (1 - self.output) * self.output
    
    def predictions(self, outputs):

        return (outputs > 0.5) * 1

class Activation_Softmax_loss_CategoricalCrossentropy:

    # not necessary with models individual instanciation of activation
    # and loss objects
    def __init__(self):

        # initialize softmax and loss
        self.activation = Activation_Softmax()
        self.loss = Loss_CategoricalCrossentropy()

    # not necessary with models individual instanciation of activation
    # and loss objects
    def forward(self, inputs, y_true, training):

        # calculate softmax on inputs 
        self.activation.forward(inputs)
        self.output = self.activation.output

        # calculate loss on softmax vs targets 
        return self.loss.calculate(self.output, y_true)
    
    def backward(self, dOdA, y_true):

        num_samples = len(dOdA)

        # convert to sparse 
        if len(y_true.shape) == 2:
            y_true = np.argmax(y_true, axis=1, keepdims=False)
        
        # gradient
        self.dOdI = dOdA.copy() 
        self.dOdI[range(num_samples), y_true] -= 1

        # normalize
        self.dOdI = self.dOdI / num_samples 
