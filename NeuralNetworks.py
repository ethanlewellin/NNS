
### Layers

#Dense Layer
class Layer_Dense: #Completely Random Dense Layer
    def __init__(self, n_inputs, n_neurons):
        self.weights = 0.01 * np.random.randn(n_inputs, n_neurons) #initialize weights
        #Note: Multiplied by 0.01 since it is often better to have start weights that minimally affect the training
        self.biases = np.zeros((1, n_neurons)) # initialize biases to 0
        #Note: initial bias for 0 is common to ensure neuron fires 
    
    def forward(self, inputs):
        self.output = np.dot(inputs, self.weights) + self.biases
        

### Activation Functions

#Relu Activation
class Activation_ReLU:
    def forward(self, inputs):
        self.output = np.maximum(0,inputs)
        
#Softmax Activation 
class Activation_Softmax:
    def forward(self, inputs):
        #Get unnormalized probabilities
        exp_values = np.exp(inputs - np.max(inputs, axis=1, keepdims = True))
        # Normalize them for each sample
        probabilites = exp_values / np.sum(exp_values, axis = 1, keepdims=True)
        
        self.output = probabilites

### Loss Functions

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