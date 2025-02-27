"""""""""""""""""""""""""""""""""""
Imports
"""""""""""""""""""""""""""""""""""
import numpy as np
import matplotlib.pyplot as plt




"""""""""""""""""""""""""""""""""""
Layers
"""""""""""""""""""""""""""""""""""

#Dense Layer
class Layer_Dense: #Completely Random Dense Layer
    # Layer initialization
    def __init__(self, n_inputs, n_neurons):
        self.weights = 0.01 * np.random.randn(n_inputs, n_neurons) #initialize weights
        #Note: Multiplied by 0.01 since it is often better to have start weights that minimally affect the training
        self.biases = np.zeros((1, n_neurons)) # initialize biases to 0
        #Note: initial bias for 0 is common to ensure neuron fires 
    
    #Forward pass
    def forward(self, inputs):
        self.inputs = inputs
        self.output = np.dot(inputs, self.weights) + self.biases
        
    #Backward Pass
    def backward(self, dvalues):
        # Gradients on parameters
        self.dweights = np.dot(self.inputs.T, dvalues)
        self.dbiases = np.sum(dvalues, axis=0, keepdims=True)
        # Gradient on values
        self.dinputs = np.dot(dvalues, self.weights.T)
        

"""""""""""""""""""""""""""""""""""
Activation Functions
"""""""""""""""""""""""""""""""""""

#Relu Activation
class Activation_ReLU:
    
    # Forward Pass
    def forward(self, inputs):
        self.inputs = inputs
        self.output = np.maximum(0,inputs)
        
    # Backward Pass
    def backward(self, dvalues):
        self.dinputs = dvalues.copy() # don't want to modify original values
        # Zero gradient where input values were negative
        self.dinputs[self.inputs <= 0] = 0
        
#Softmax Activation 
class Activation_Softmax:
    def forward(self, inputs):
        # Remember input values
        self.inpus = inputs
        #Get unnormalized probabilities
        exp_values = np.exp(inputs - np.max(inputs, axis=1, keepdims = True))
        # Normalize them for each sample
        probabilites = exp_values / np.sum(exp_values, axis = 1, keepdims=True)
        
        self.output = probabilites
    
    def backward(self, dvalues):    
        # Create uninitialized array
        self.dinputs=np.empty_like(dvalues)
        
        # Enumerate outputs and gradients
        for index, (single_output, single_dvalues) in enumerate(zip(self.output, dvalues)):
            #Flatten output array
            single_output = single_output.reshape(-1,1)
            #Calculate Jacobian matrix of the output
            jacobian_matrix = np.diagflat(single_output) - np.dot(single_output, single_output.T)
            #Calculate sample-wise gradient and add it to the array of sample gradients
            self.dinputs[index] = np.dot(jacobian_matrix, single_dvalues)

"""""""""""""""""""""""""""""""""""
Loss Functions
"""""""""""""""""""""""""""""""""""

#Common Loss Class
class Loss:
    def calculate(self, output, y):
        #calculate sample losses
        sample_losses = self.forward(output,y)
        # Calculate mean loss
        data_loss = np.mean(sample_losses)
        
        return data_loss
    
#Cross-entropy loss
class Loss_CategoricalCrossentropy(Loss):

    #Forward Pass
    def forward(self, y_pred, y_true):
            
        #Number of samples in a batch
        samples = len(y_pred)
            
        # Clip data to prevent division by 0
        # Clip both sides to not affect mean
        y_pred_clipped = np.clip(y_pred, 1e-7, 1- 1e-7)
           
        # Probabilities for target values
        # only if categorical labels
        if len(y_true.shape)==1:
            correct_confidences = y_pred_clipped[
                range(samples),
                y_true
            ]
                
        # Mask values - only for one-hot encoded labels
        elif len(y_true.shape) == 2:
            correct_confidences = np.sum(
                y_pred_clipped * y_true,
                axis = 1
            )
                
        #Losses
        negative_log_likelihoods = -np.log(correct_confidences)
        return negative_log_likelihoods
        
     # Backward pass
    def backward ( self , dvalues , y_true ):
        
        # Number of samples
        samples = len (dvalues)
        # Number of labels in every sample
        # We'll use the first sample to count them
        labels = len (dvalues[ 0 ])
        
        # If labels are sparse, turn them into one-hot vector
        if len (y_true.shape) == 1 :
            y_true = np.eye(labels)[y_true]
            
        # Calculate gradient
        self.dinputs = - y_true / dvalues
        # Normalize gradient
        self.dinputs = self.dinputs / samples
        
        
"""""""""""""""""""""""""""""""""""
Combined Activation and Loss Functions
"""""""""""""""""""""""""""""""""""
# Softmax classifier - combined Softmax activation
# and cross-entropy loss for faster backward step
class Activation_Softmax_Loss_CategoricalCrossentropy ():
    # Creates activation and loss function objects
    def __init__ ( self ):
        self.activation = Activation_Softmax()
        self.loss = Loss_CategoricalCrossentropy()
    # Forward pass
    def forward ( self , inputs , y_true ):
        # Output layer's activation function
        self.activation.forward(inputs)
        # Set the output
        self.output = self.activation.output
        # Calculate and return loss value
        return self.loss.calculate(self.output, y_true)
    # Backward pass
    def backward ( self , dvalues , y_true ):
        # Number of samples
        samples = len (dvalues)
        # If labels are one-hot encoded,
        # turn them into discrete values
        if len (y_true.shape) == 2 :
            y_true = np.argmax(y_true, axis = 1 )
        # Copy so we can safely modify
        self.dinputs = dvalues.copy()
        # Calculate gradient
        self.dinputs[ range (samples), y_true] -= 1
        # Normalize gradient
        self.dinputs = self.dinputs / samples