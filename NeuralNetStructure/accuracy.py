import numpy as np

PARAMETER = 250

class Accuracy:

    def new_pass(self):
        self.accumulated_sum = 0
        self.accumulated_count = 0

    def calculate_accumulated(self):
        # epoch wise data
        accuracy = self.accumulated_sum / self.accumulated_count

        return accuracy 

    def calculate(self, predictions, y):
        # predictions is determined from output_layer_activation, formatted as to give values for activated neurons and off neurons
        comparisons = self.compare(predictions, y)

        accuracy = np.mean(comparisons)

        self.accumulated_sum += np.sum(comparisons)
        if comparisons.shape == 2:
            self.accumulated_count += len(np.concatenate(comparisons))
        else:
            self.accumulated_count += len(comparisons)
 
        return accuracy 

class Accuracy_Regression(Accuracy):

    # predicting continuous data

    def __init__(self):
        self.precision = None
    
    def init(self, y, reinit=False):
        if self.precision is None or reinit:
            self.precision = np.std(y) / PARAMETER

    def compare(self, predictions, y):
        return np.absolute(predictions - y) < self.precision

class Accuracy_Categorical(Accuracy):

    # predicting discrete data

    def init(self, y):
        pass
    
    def compare(self, predictions, y):
        # Predictions are already in max neuron activation format 
        maxes = y
        if len(y.shape) == 2:
            maxes = np.argmax(y, axis=1, keepdims=False)
        
        return predictions == maxes

class Accuracy_Binary(Accuracy):
    def init(self, y):
        pass
    def compare(self, predictions, y):

        return predictions == y


