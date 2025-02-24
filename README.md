### Very Awesome Neural Network

# How does it work ⁉️

## Brief summary

### Structure

A neural network is composed of nodes with edge weights. 
- The **nodes** represent individual computations that take inputs and apply a function, and produce an output.
    - Nodes are seperated into vertically aligned layers. 
        - In an input layer, nodes represent features of the input data. 
        - In hidden layers, nodes represent learned abstract features.
        - In the output layer, nodes represent final predictions or classifications.
- The **edge** weights represent the importance of the input being passed to the node (usually referred to as the 'strength of connections between nodes'). 
    - Higher weights mean stronger influence on the next neuron.
    - Weights are adjusted during training to minimise the error in predictions (using backpropagation and gradient descent).

### Training process

The neural network continuously improves through an iterative process of forward propagation and backpropagation.
- **Forward propagation**: The network processes input data to make predictions. The difference between predictions and actual values gives an error.
- **Backpropagation**: The network reduces error by adjusting weights and biases.
    - This is done using the error function to compute gradients, and gradient descent to update weights in the optimal direction.

# The formulae involved

## Forward Propagation

### Hidden Layer

#### Pre-activation (Linear Transformation) 
$$Z^{[1]} = W^{[1]}X + b^{[1]}$$

- Symbols
    - $X$: the input data.
    - $W^{[1]}$: Weights matrix for the hidden layer.
    - $b^{[1]}$: Bias vector for the hidden layer. 

#### Activation (ReLU function)
$$A^{[1]} = ReLU(Z^{[1]})$$

- Applying the activation function gives us the final, non-linear neuron output. This determines how much influence it will have on the next layer.
- We need the neuron output to be non-linear as otherwise our neural network is essentially just a linear transformation. A linear transformation is not sufficient because it won't be able to 'learn' complex features.
- The activation function $ReLU$ outputs the input directly if it is positive, otherwise it outputs 0.

### Output Layer

#### Pre-activation (Linear Transformation)
$$Z^{[2]} = W^{[2]}X + b^{[2]}$$

#### Activation (Softmax function)
$$A^{[2]} = Softmax(Z^{[2]})$$

Where the Softmax formula for each element is 
$$A_i^{[2]} = \frac{e^{Z_i^{[2]}}}{\sum_j e^{Z_j^{[2]}}}$$

- The $Softmax$ function converts the raw scores $Z^{[2]}$ into probabilities that sum to 1 across the output classes.

## Compute Loss Function

$$ J = -\frac{1}{m} \sum ^m _{i=1} \sum ^{n^{[2]}}_{j=1} Y_j ^{(i)} \log (A_j ^{[2](i)}) $$

- Measures the difference between the predicted outputs $A^{[2]}$ and the true labels $Y$.
- Symbols 
    - $m$: Number of training examples.
    - $n$: Number of output neurons.
    - $Y_j ^{(i)}$: True label (we will one-hot encode this) for the $j$-th outut for the $i$-th training example.
    - $A_j ^{[2](i)}$: Predicted probability (from softmax) for the $j$-th output of the $i$-th training example.

## Backpropagation (Computing Gradients)

- Compute the gradients for the loss function with respect to each parameter.

### Output Layer
#### Gradient of $Z^{[2]}$ (Error at Output Layer):

$$dZ^{[2]} = A^{[2]} - Y$$

- Represents how much the predicted probabilities deviate from the true labels.

#### Gradient of Weights $W^{[2]}$:

$$dW^{[2]} = \frac{1}{m} dZ^{[2]}A^{[1]T}$$

- Gradient of the loss with respect to the output layer weights.

#### Gradient of Biases $b^{[2]}$:

$$db^{[2]} = \frac{1}{m} \sum dZ^{[2]}$$

- Gradient of the loss with respect to the output layer biases.

### Hidden Layer

#### Gradient of $Z^{[1]}$ (Error Propagation to Previous Layer)

$$dZ^{[1]} = W^{[2]T}dZ^{[2]}\cdot ReLU'(Z^{[1]})$$

- Error signal for the hidden layer.
- The element-wise multiplication of the Hadamard product ($\cdot$) ensures that only active neurons receive gradients (inactive gradients have value 0).

> #### Derivation
>
> Apply the chain rule
>
> $$dZ^{[1]} = \frac{\partial J}{\partial Z^{[1]}} = \frac{\partial J}{\partial A^{[1]}} \cdot \frac{\partial A^{[1]}}{\partial Z^{[1]}}$$
>
> Since $J$ is a function of $Z^{[2]}$ and $Z^{[2]}$ is a function of $A^{[1]}$, the chain rule tells us:
> $$\frac{\partial J}{\partial A^{[1]}} = \frac{\partial J}{\partial Z^{[2]}} \cdot \frac{\partial Z^{[2]}}{\partial A^{[1]}}$$
>
> We can then compute $\frac{\partial Z^{[2]}}{\partial A^{[1]}}$. 
>
> Since
> $$Z^{[2]} = W^{[2]} A^{[1]} + b^{[2]}, $$
> this relationship is linear in $A^{[1]}$. 
> Therefore, taking the derivative with respect to $A^{[1]}$ gives:
> $$\frac{\partial Z^{[2]}}{\partial A^{[1]}} = W^{[2]}$$
> Therefore, from our chain rule expression:
> $$\frac{\partial J}{\partial A^{[1]}} = dZ^{[2]} \cdot W^{[2]}$$
> And to account for the dimensions and order of multiplication (recall the rules of matrix multiplication), the proper matrix product is:
> $$\frac{\partial J}{\partial A^{[1]}} = W^{[2]T} dZ^{[2]}$$
> Then, recall that 
> $$A^{[1]} = ReLU(Z^{[1]})$$
> which means that we have the simple result 
> $$\frac{\partial A^{[1]}}{\partial Z^{[1]}} = ReLU'(Z^{[1]})$$
> And so finally we have:
> $$dZ^{[1]} = \frac{\partial J}{\partial Z^{[1]}} = \frac{\partial J}{\partial A^{[1]}} \cdot \frac{\partial A^{[1]}}{\partial Z^{[1]}} = W^{[2]T} dZ^{[2]} \odot ReLU'(Z^{[1]})$$
> Where $\odot$ refers to the Hadamard product, which is element-wise multiplication.

#### Gradient of Weights $W^{[1]}$:

$$dW^{[1]} = \frac{1}{m} dZ^{[1]} X^T$$

- Gradient of the loss with respect to the hidden layer weights.

#### Gradient of Biases $b^{[1]}$:

$$db^{[1]} = \frac{1}{m} \sum dZ ^{[1]}$$

- Gradient of the loss with respect to the hidden layer biases.

## Paramterer Updates (Gradient Descent)

$$W^{[l]} = W^{[l]} - \alpha dW^{[l]}$$

$$b^{[l]} = b^{[l]} - \alpha db^{[l]}$$

- Where $\alpha$ is the learning rate (hyperparameter for step size).

# About this dataset

## Aim

Make a neural network that can recognise handwritten numbers from 0-9. That is, using a `28 * 28` handwritten image, correctly classify the image as number from 0-9.

### Training data

The dataset comes from the MNIST database, a pretty well known collection of handwritten digits from 0 to 9. Once trained the neural network should be able to recognise handwritten numbers it has never seen before.