### Very Awesome Neural Network

## How does it work ⁉️

### Brief summary

#### Structure

A neural network is composed of nodes with edge weights. 
- The **nodes** represent individual computations that take inputs and apply a function, and produce an output.
    - Nodes are seperated into vertically aligned layers. 
        - In an input layer, nodes represent features of the input data. 
        - In hidden layers, nodes represent learned abstract features.
        - In the output layer, nodes represent final predictions or classifications.
- The **edge** weights represent the importance of the input being passed to the node (usually referred to as the 'strength of connections between nodes'). 
    - Higher weights mean stronger influence on the next neuron.
    - Weights are adjusted during training to minimise the error in predictions (using backpropagation and gradient descent).

#### Training process

The neural network continuously improves through an iterative process of forward propagation and backpropagation.
- **Forward propagation**: The network processes input data to make predictions. The difference between predictions and actual values gives an error.
- **Backpropagation**: The network reduces error by adjusting weights and biases.
    - This is done using the error function to compute gradients, and gradient descent to update weights in the optimal direction.

## About this dataset

### Aim

Make a neural network that can recognise handwritten numbers from 0-9. That is, using a `28 * 28` handwritten image, correctly classify the image as number from 0-9.

### Training data

The dataset comes from the MNIST database, a pretty well known collection of handwritten digits from 0 to 9. Once trained the neural network should be able to recognise handwritten numbers it has never seen before.