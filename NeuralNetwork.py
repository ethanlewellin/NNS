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
    def __init__(self, n_inputs, n_neurons,
                 weight_regularizer_l1 = 0 , weight_regularizer_l2 = 0 ,
                 bias_regularizer_l1 = 0 , bias_regularizer_l2 = 0 ):
        
        self.weights = 0.01 * np.random.randn(n_inputs, n_neurons) #initialize weights
        #Note: Multiplied by 0.01 since it is often better to have start weights that minimally affect the training
        self.biases = np.zeros((1, n_neurons)) # initialize biases to 0
        #Note: initial bias for 0 is common to ensure neuron fires 
        
        # Set regularization strength (lambdas)
        self.weight_regularizer_l1 = weight_regularizer_l1
        self.weight_regularizer_l2 = weight_regularizer_l2
        self.bias_regularizer_l1 = bias_regularizer_l1
        self.bias_regularizer_l2 = bias_regularizer_l2
    
    #Forward pass
    def forward(self, inputs):
        self.inputs = inputs
        self.output = np.dot(inputs, self.weights) + self.biases
        
    #Backward Pass
    def backward(self, dvalues):
        # Gradients on parameters
        self.dweights = np.dot(self.inputs.T, dvalues)
        self.dbiases = np.sum(dvalues, axis=0, keepdims=True)
        
        # Gradient on vregularization
        # L1 on weights
        if self.weight_regularizer_l1 > 0:
            dL1 = np.ones_like(self.weights)
            dL1[self.weights < 0 ] = - 1
            self.dweights += self.weight_regularizer_l1 * dL1
        # L2 on weights
        if self.weight_regularizer_l2 > 0 :
            self.dweights += 2 * self.weight_regularizer_l2 * self.weights
        # L1 on biases
        if self.bias_regularizer_l1 > 0 :
            dL1 = np.ones_like(self.biases)
            dL1[self.biases < 0 ] = - 1
            self.dbiases += self.bias_regularizer_l1 * dL1
        # L2 on biases
        if self.bias_regularizer_l2 > 0 :
            self.dbiases += 2 * self.bias_regularizer_l2 * self.biases
        
        # Gradient on values
        self.dinputs = np.dot(dvalues, self.weights.T)

#Dropout layer
class Layer_Dropout:
    
    # Init
    def __init__(self, rate):
        # Store rate, we invert it for use in the binomial distribution
        self.rate = 1 - rate
        
    # Forward Pass
    def forward(self, inputs):
        # Save input values
        self.inputs = inputs
        # Generate and save scaled mask
        self.binary_mask = np.random.binomial(1, self.rate, size = inputs.shape) / self.rate
        # Apply mask to output values
        self.output = inputs * self.binary_mask
        
    # Backward Pass
    def backward(self, dvalues):
        # Gradient on values
        self.dinputs = dvalues * self.binary_mask
        

"""""""""""""""""""""""""""""""""""
Activation Functions
"""""""""""""""""""""""""""""""""""

#Relu Activation
## On/off linear function, easy to optimize
## Most popular, the "go-to" function
## Can cause dying neurons 
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
## Typically used in the last hidden layer
## Calculates the probabilty distribution over 'n' different events
## Dependant probabilites, sum of proabilites  = 1
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
            
# Sigmoid Activation
## Creates values between 0 and 1
## Difficult optimization becuase it is not 0-centered  
## Good for classification
## Independant probabilities =, sum of probabilities not necessarily equal to 1
class Activate_Sigmoid:
    
    #Forward Pass
    def forward(self, inputs):
        # Save input and calculate/save output of sigmoid function
        self.inputs = inputs
        self.output = 1/(1+np.exp(-inputs))
            
    # Backward pass
    def backward(self, dvalues):
        # Derivative - calculates from output of the sigmoid function
        self.dinputs = dvalues * (1 - self.output) * self.output
        
    
# Linear activation function
# Most basic regression activation function    
class Activation_Linear:
    
    # Forward pass
    def forward(self, inputs):
        # Just remember values
        self.inputs = inputs
        self.output = inputs
        
    # Backward Pass
    def backward(self, dvalues):
        # deriviative of linear function is 1
        self.dinputs = dvalues.copy()

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
    
    # Regularization loss calculation
    def regularization_loss(self, layer):
        
        # 0 by default
        regularization_loss = 0
        
        # L1 regularization - weights
        if layer.weight_regularizer_l1 > 0:
            regularization_loss += layer.weight_regularizer_l1 * np.sum(np.abs(layer.weights))
        
        # L2 regularization - weights
        if layer.weight_regularizer_l2 > 0:
            regularization_loss += layer.weight_regularizer_l2 * np.sum(layer.weights * layer.weights)
            
        # L1 regularization - biases
        if layer.bias_regularizer_l1 > 0:
            regularization_loss += layer.bias_regularizer_l1 * np.sum(np.abs(layer.biases))
            
        # L2 regularization - biases
        if layer.bias_regularizer_l2 > 0:
            regularization_loss += layer.bias_regularizer_l2 * np.sum(layer.biases * layer.biases)
        
        return regularization_loss
    
# Categorical Cross-entropy loss
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
        
# Binary Cross-entropy loss
class Loss_BinaryCrossentropy(Loss):
    
    # Forward Pass
    def forward(self, y_pred, y_true):
        # Clip data to prevent division by 0
        # Clip both sides to not drag mean towards any value
        y_pred_clipped = np.clip(y_pred, 1e-7 , 1 - 1e-7 )
        
        # Calculate sample-wise loss
        sample_losses = -(y_true * np.log(y_pred_clipped) +
                          (1 - y_true) * np.log(1 - y_pred_clipped))
        sample_losses = np.mean(sample_losses, axis=-1)

        return sample_losses
    
    #Backward Pass
    def backward(self, dvalues, y_true):
        
        # Number of samples
        samples = len(dvalues)
        # Number of outputs in every sample
        outputs = len(dvalues[0])
        
        # Clip the data to prevent division by 0
        clipped_dvalues = np.clip(dvalues, 1e-7 , 1 - 1e-7 )
        
        # Calculate gradient
        self.dinputs = -(y_true / clipped_dvalues - 
                        (1 - y_true) / (1 - clipped_dvalues)) / outputs
        # Normailize gradient
        self.dinputs = self.dinputs / samples
    
   
# Mean Squared Error (L2) Loss
class Loss_MeanSquaredError(Loss): # L2 loss
    
    # Forward Pass
    def forward(self, y_pred, y_true):
        
        # Calculate loss
        sample_losses = np.mean((y_true - y_pred)**2, axis=-1)
        return sample_losses
    
    def backward(self, dvalues, y_true):
        
        # Number of Samples
        samples = len(dvalues)
        outputs = len(dvalues[0])
        
        #Gradient on values
        self.dinputs = -2 * (y_true - dvalues) / outputs
        # Normalize Gradient
        self.dinputs = self.dinputs / samples
        
# Mean Absolute Error (L1) Loss        
class Loss_MeanAbsoluteError(Loss): # L1 loss
    
    # Forward Pass
    def forward(self, y_pred, y_true):
        
        # Calculate loss
        sample_losses = np.mean(np.abs(y_true - y_pred), axis = -1)
        
        # Return losses
        return sample_losses
    
    # Backward pass
    def backward(self, dvalues, y_true):
        
        # Number of samples
        samples = len(dvalues)
        outputs = len(dvalues[0])
        
        # Calculate Gradient
        self.dinputs = np.sign(y_true - dvalues) / outputs
        # Normalize Gradient
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
        
        
"""""""""""""""""""""""""""""""""""
Optimizer Functions
"""""""""""""""""""""""""""""""""""

class Optimizer_SGD :
    
    # Initialize optimizer - set settings,
    # learning rate of 1. is default for this optimizer
    def __init__ ( self , learning_rate = 1. , decay = 0. , momentum = 0. ):
        self.learning_rate = learning_rate
        self.current_learning_rate = learning_rate
        self.decay = decay
        self.iterations = 0
        self.momentum = momentum
        
    # Call once before any parameter updates
    def pre_update_params ( self ):
        if self.decay:
            self.current_learning_rate = self.learning_rate * ( 1. / ( 1. + self.decay * self.iterations))
            
    # Update parameters
    def update_params ( self , layer ):
        
        # If we use momentum
        if self.momentum:
            # If layer does not contain momentum arrays, create them filled with zeros
            if not hasattr (layer, 'weight_momentums' ):
                layer.weight_momentums = np.zeros_like(layer.weights)
                # If there is no momentum array for weights
                # The array doesn't exist for biases yet either.
                layer.bias_momentums = np.zeros_like(layer.biases)
                
            # Build weight updates with momentum - take previous
            # updates multiplied by retain factor and update with
            # current gradients
            weight_updates = \
                self.momentum * layer.weight_momentums - \
                self.current_learning_rate * layer.dweights
            layer.weight_momentums = weight_updates
            
            # Build bias updates
            bias_updates = \
                self.momentum * layer.bias_momentums - \
                self.current_learning_rate * layer.dbiases
            layer.bias_momentums = bias_updates
            
        # Vanilla SGD updates (as before momentum update)
        else :
            weight_updates = - self.current_learning_rate * layer.dweights
            bias_updates = - self.current_learning_rate * layer.dbiases
            
        # Update weights and biases using either
        # vanilla or momentum updates
        layer.weights += weight_updates
        layer.biases += bias_updates
    
    # Call once after any parameter updates
    def post_update_params ( self ):
        self.iterations += 1

# AdaGrad Optimizer
class Optimizer_Adagrad :
    
    # Initialize optimizer - set settings
    def __init__ ( self , learning_rate = 1. , decay = 0. , epsilon = 1e-7 ):
        self.learning_rate = learning_rate
        self.current_learning_rate = learning_rate
        self.decay = decay
        self.iterations = 0
        self.epsilon = epsilon
        
    # Call once before any parameter updates
    def pre_update_params ( self ):
        if self.decay:
            self.current_learning_rate = self.learning_rate * ( 1. / ( 1. + self.decay * self.iterations))
 
    # Update parameters
    def update_params ( self , layer ):
        
        # If layer does not contain cache arrays,
        # create them filled with zeros
        if not hasattr (layer, 'weight_cache' ):
            layer.weight_cache = np.zeros_like(layer.weights)
            layer.bias_cache = np.zeros_like(layer.biases)
        
        # Update cache with squared current gradients
        layer.weight_cache += layer.dweights ** 2
        layer.bias_cache += layer.dbiases ** 2
        
        # Vanilla SGD parameter update + normalization
        # with square rooted cache
        layer.weights += - self.current_learning_rate * \
                layer.dweights / \
                (np.sqrt(layer.weight_cache) + self.epsilon)
        layer.biases += - self.current_learning_rate * \
            layer.dbiases / \
                (np.sqrt(layer.bias_cache) + self.epsilon)
                
    # Call once after any parameter updates
    def post_update_params ( self ):
        self.iterations += 1
        
# RMSprop optimizer
class Optimizer_RMSprop :
    
    # Initialize optimizer - set settings
    def __init__ ( self , learning_rate = 0.001 , decay = 0. , epsilon = 1e-7 , rho = 0.9 ):
        self.learning_rate = learning_rate
        self.current_learning_rate = learning_rate
        self.decay = decay
        self.iterations = 0
        self.epsilon = epsilon
        self.rho = rho
        
    # Call once before any parameter updates
    def pre_update_params ( self ):
        if self.decay:
            self.current_learning_rate = self.learning_rate * ( 1. / ( 1. + self.decay * self.iterations))
    
    # Update parameters
    def update_params ( self , layer ):
        
        # If layer does not contain cache arrays,
        # create them filled with zeros
        if not hasattr (layer, 'weight_cache' ):
            layer.weight_cache = np.zeros_like(layer.weights)
            layer.bias_cache = np.zeros_like(layer.biases)
            
        # Update cache with squared current gradients
        layer.weight_cache = self.rho * layer.weight_cache + ( 1 - self.rho) * layer.dweights ** 2
        layer.bias_cache = self.rho * layer.bias_cache + ( 1 - self.rho) * layer.dbiases ** 2

        # Vanilla SGD parameter update + normalization
        # with square rooted cache
        layer.weights += - self.current_learning_rate * layer.dweights / \
            (np.sqrt(layer.weight_cache) + self.epsilon)
        layer.biases += - self.current_learning_rate * \
            layer.dbiases / \
                (np.sqrt(layer.bias_cache) + self.epsilon)
                
    # Call once after any parameter updates
    def post_update_params ( self ):
        self.iterations += 1

# Adam Optimizer
class Optimizer_Adam :
    
    # Initialize optimizer - set settings
    def __init__ ( self , learning_rate = 0.001 , decay = 0. , epsilon = 1e-7 , beta_1 = 0.9 , beta_2 = 0.999 ):
        self.learning_rate = learning_rate
        self.current_learning_rate = learning_rate
        self.decay = decay
        self.iterations = 0
        self.epsilon = epsilon
        self.beta_1 = beta_1
        self.beta_2 = beta_2

    # Call once before any parameter updates
    def pre_update_params ( self ):
        if self.decay:
            self.current_learning_rate = self.learning_rate * ( 1. / ( 1. + self.decay * self.iterations))
    
    # Update parameters
    def update_params ( self , layer ):
        # If layer does not contain cache arrays,
        # create them filled with zeros
        if not hasattr (layer, 'weight_cache' ):
            layer.weight_momentums = np.zeros_like(layer.weights)
            layer.weight_cache = np.zeros_like(layer.weights)
            layer.bias_momentums = np.zeros_like(layer.biases)
            layer.bias_cache = np.zeros_like(layer.biases)
            
        # Update momentum with current gradients
        layer.weight_momentums = self.beta_1 * \
            layer.weight_momentums + \
                ( 1 - self.beta_1) * layer.dweights
                
        layer.bias_momentums = self.beta_1 * \
            layer.bias_momentums + \
                ( 1 - self.beta_1) * layer.dbiases
                
        # Get corrected momentum
        # self.iteration is 0 at first pass
        # and we need to start with 1 here
        weight_momentums_corrected = layer.weight_momentums / \
            ( 1 - self.beta_1 ** (self.iterations + 1 ))
            
        bias_momentums_corrected = layer.bias_momentums / \
            ( 1 - self.beta_1 ** (self.iterations + 1 ))
            
        # Update cache with squared current gradients
        layer.weight_cache = self.beta_2 * layer.weight_cache + \
            ( 1 - self.beta_2) * layer.dweights ** 2
            
        layer.bias_cache = self.beta_2 * layer.bias_cache + \
            ( 1 - self.beta_2) * layer.dbiases ** 2
            
        # Get corrected cache
        weight_cache_corrected = layer.weight_cache / \
            ( 1 - self.beta_2 ** (self.iterations + 1 ))
            
        bias_cache_corrected = layer.bias_cache / \
            ( 1 - self.beta_2 ** (self.iterations + 1 ))
            
        # Vanilla SGD parameter update + normalization
        # with square rooted cache
        layer.weights += - self.current_learning_rate * \
            weight_momentums_corrected / \
                (np.sqrt(weight_cache_corrected) + self.epsilon)
                
        layer.biases += - self.current_learning_rate * \
            bias_momentums_corrected / \
                (np.sqrt(bias_cache_corrected) + self.epsilon)
                
                
    # Call once after any parameter updates
    def post_update_params ( self ):
        self.iterations += 1
        