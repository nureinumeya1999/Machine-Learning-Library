from NeuralNetStructure.accuracy import *
from NeuralNetStructure.activation import *
from NeuralNetStructure.dropout import *
from NeuralNetStructure.layer import * 
from NeuralNetStructure.loss import *
from NeuralNetStructure.optimizers import * 
import os
import pickle
import copy

class Model:

    def __init__(self, type="1D", resolution=None):

        if resolution is not None:
            self.resolution = resolution

        self.type = type
        self.dense_layers = []

        if type == "2D":
            self.conv_dims = [(1, self.resolution[0], self.resolution[1])]
            self.conv_layers = []

        self.layers = []

        self.softmax_classifier_output = None

    def add(self, layer, layer_type=None):

        if layer_type == "DENSE":
            self.dense_layers.append(layer)
        elif layer_type == "CONV":
            self.conv_layers.append(layer)

            curr_dims = self.conv_dims[-1]
            if hasattr(layer, "update_dims"):
                self.conv_dims.append(layer.update_dims(curr_dims))
        
        self.layers.append(layer)

    def set(self, *, loss, optimizer=None, accuracy):
        
        self.loss = loss
        self.optimizer = optimizer
        if accuracy is not None:
            self.accuracy = accuracy 
        
    
    def train(self, X, y, *, epochs=1, batch_size=None, print_every=1, validation_data=None):

        # if batch_size is None, operate on full dataset
        train_steps = 1
        if validation_data is not None:
            validation_steps = 1
            X_val, y_val = validation_data
        
        if batch_size is not None:

            train_steps = len(X) // batch_size
            if train_steps * batch_size < len(X):
                train_steps += 1

            if validation_data is not None:
                validation_steps = len(X_val) // batch_size
                if validation_steps * batch_size < len(X_val):
                    validation_steps += 1


        self.accuracy.init(y)
        
        for epoch in range(1,  epochs+1):
            print(f'epoch: {epoch}')

            # reset epoch wise metrics for new training pass
            self.loss.new_pass()
            self.accuracy.new_pass()

            for step in range(train_steps):
                
                #initialize batch
                if batch_size is None:
                    batch_X = X
                    batch_y = y
                else:
                    batch_X = X[step * batch_size :
                                                (step + 1) * batch_size]
                    batch_y = y[step * batch_size :
                                                (step + 1) * batch_size]
            
                # calculate output after full forward pass for metrics
            
                output = self.forward(batch_X, training=True) 
                
                # step wise loss and accuracy
                data_loss, regularization_loss = self.loss.calculate(output, batch_y, include_regularization=True)
                loss = data_loss + regularization_loss
                predictions = self.output_layer_activation.predictions(output)
                accuracy = self.accuracy.calculate(predictions, batch_y)
            
                # calculate gradient after backward pass
                self.backward(output, batch_y)


                # optimize in direction of negative gradient
                self.optimizer.pre_update_params()
                for layer in self.trainable_layers:
                    self.optimizer.update_params(layer)
                self.optimizer.post_update_params()

                # display step wise metrics
                if not step % print_every or step == train_steps - 1:
                    print(
                    f'step: {step}, ' + 
                    f'acc: {accuracy:.3f}, ' + 
                    f'loss: {loss:.3f}, ' +
                    f'data_loss: {data_loss:.3f}, ' +
                    f'reg_loss: {regularization_loss:.3f}, '+
                    f'lr: {self.optimizer.current_learning_rate}')
            
            # epoch wise loss and accuracy 
            epoch_data_loss, epoch_regularization_loss = self.loss.calculate_accumulated(include_regularization=True)
            epoch_loss = epoch_data_loss + epoch_regularization_loss
            epoch_accuracy = self.accuracy.calculate_accumulated()

            # display epoch wise metrics
            print(f'training, ' + 
                    f'acc: {epoch_accuracy:.3f}, ' + 
                    f'loss: {epoch_loss:.3f}, ' +
                    f'data_loss: {epoch_data_loss:.3f}, ' +
                    f'reg_loss: {epoch_regularization_loss:.3f}, '+
                    f'lr: {self.optimizer.current_learning_rate}')

            # reset epoch wise metrics for new validation pass
            # should there be validation data
            if validation_data is not None:
                self.evaluate(validation_data[0], validation_data[1], batch_size=batch_size)


    def finalize(self):

        self.layer_count = len(self.layers)

        self.trainable_layers = []

        self.input_layer = Layer_Input()
        
        for i in range(self.layer_count):

            if i == 0:
                self.layers[i].prev = self.input_layer
                self.layers[i].next = self.layers[i + 1]

            elif i < self.layer_count - 1:
                self.layers[i].prev = self.layers[i - 1]
                self.layers[i].next = self.layers[i + 1]
            
            else:
                self.layers[i].prev = self.layers[i - 1]
                self.layers[i].next = self.loss

                self.output_layer_activation = self.layers[i]

            if hasattr(self.layers[i], "weights"):
                self.trainable_layers.append(self.layers[i])
            
        if self.loss is not None:    
            self.loss.remember_trainable_layers(self.trainable_layers)   

        if isinstance(self.layers[-1], Activation_Softmax) and \
            isinstance(self.loss, Loss_CategoricalCrossentropy):
            self.softmax_classifier_output = Activation_Softmax_loss_CategoricalCrossentropy()


    def forward(self, X, training):

        self.layers[0].forward(X, training)

        for layer in self.layers[1:]:
            layer.forward(layer.prev.output, training)

        return layer.output
 
    def backward(self, output, y):
        
        if self.softmax_classifier_output is not None:
            self.softmax_classifier_output.backward(output, y)
            self.layers[-1].dOdI = self.softmax_classifier_output.dOdI
            for layer in reversed(self.layers[:-1]):
                layer.backward(layer.next.dOdI)
                
            return


        self.loss.backward(output, y)


        for layer in reversed(self.layers):

            layer.backward(layer.next.dOdI)


    def evaluate(self, X_val, y_val, *, batch_size=None):
        validation_steps = 1
        if batch_size is not None:
            validation_steps = len(X_val) // batch_size
            if validation_steps * batch_size < len(X_val):
                validation_steps += 1
    
        self.loss.new_pass()
        self.accuracy.new_pass()

        for step in range(validation_steps):

            #initialize batch
            if batch_size is None:
                batch_X = X_val
                batch_y = y_val

            else:
                batch_X = X_val[step * batch_size: (step + 1) * batch_size]
                batch_y = y_val[step * batch_size : (step + 1) * batch_size]


            # full forward pass to calculate metrics
            output = self.forward(batch_X, training=False)
            
            # step wise loss and accuracy 
            self.loss.calculate(output, batch_y)

            predictions = self.output_layer_activation.predictions(output)
            self.accuracy.calculate(predictions, batch_y)

        # epoch wise loss and accuracy
        validation_loss = self.loss.calculate_accumulated()
        validation_accuracy = self.accuracy.calculate_accumulated()

        print(f'validation, ' + f'acc: {validation_accuracy:.3f}, ' 
        + f'loss: {validation_loss:.3f}')



#========== SAVE/LOAD TASKS ===========================================================

    def get_parameters(self):
        parameters = []
        for layer in self.trainable_layers:
            parameters.append(layer.get_parameters())
        return parameters

    def set_parameters(self, parameters):
        for parameter_set, layer in zip(parameters, self.trainable_layers):
            layer.set_parameters(*parameter_set)
    
    def save_parameters(self, path):
        
        if not os.path.exists(path):
            os.mkdir(path)
        
        NNtype = "CNN" if self.type == "2D" else "DNN"
        id = 1
        while os.path.exists(os.path.join(path, NNtype + "_planes_parameters_" + str(id).zfill(4) + ".parms")):
            id += 1

        NEW_PARAMETERS_PATH = os.path.join(path, NNtype + "_planes_parameters_" + str(id).zfill(4) + ".parms")
        with open(NEW_PARAMETERS_PATH, "wb") as f:
            pickle.dump(self.get_parameters(), f)
    
    def load_parameters(self, path):

        with open(path, 'rb') as f:
            self.set_parameters(pickle.load(f))

    def save(self, path):
        model = copy.deepcopy(self)
        model.loss.new_pass()
        model.accuracy.new_pass()
        model.input_layer.__dict__.pop('output', None)
        model.loss.__dict__.pop('dOdI', None)

        for layer in model.layers:
            for property in ['inputs', 'output', 'dOdI', 'dweights', 'dbiases']:
                layer.__dict__.pop(property, None)

        with open(path, 'wb') as f:
            pickle.dump(model, f)
    
    @staticmethod
    def load(path):
        with open(path, 'rb') as f:
            model = pickle.load(f)
        return model

    
#========================================================================================


    def predict(self, X, *, batch_size=None):
            prediction_steps = 1

            if batch_size is not None:
                prediction_steps = len(X) // batch_size

                if prediction_steps * batch_size < len(X):
                    prediction_steps += 1
            output = []
            predictions = []
            for step in range(prediction_steps):
                if batch_size is None:
                    batch_X = X
                else:
                    batch_X = X[step * batch_size : (step + 1) * batch_size]

                batch_output = self.forward(batch_X, training=False)

                output.append(batch_output)
                predictions.append(self.output_layer_activation.predictions(batch_output))

            # return confidences stacked sample wise
            return self.output_layer_activation.output