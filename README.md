# Neural-Network-from-scratch

A modular neural network is a type of machine learning model that is composed of small, independent building blocks, or modules, which can be connected together in different configurations to perform different tasks. Building a modular neural network from scratch using object-oriented programming (OOP) in Python involves creating custom classes for each module in the network, such as layers, activation functions, and optimizers. These classes should be designed to interact seamlessly with one another, so that the network can be easily assembled and trained on different datasets.

The project started by defining the base Neural Network class and then a "Dense" or "Fully-Connected" type of layer. Each layer class should have its own methods for forwarding the input through the layer and updating the layer's parameters during training. Activation functions like ReLU, Sigmoid,Tanh and Softmax are also implemented as separate classes.

Additionally, the project implemented the "gradient descent" optimization algorithm to minimize the loss function which is generally a cross-entropy function or a mean squared error.

The final step of the project was to test the modular neural network on a simple dataset : the binary XOR function and evaluate its performance using mean squared error, and visualize the results or decision boundaries.

Overall, this project would have provided an understanding about the fundamental concepts behind neural networks and the implementation of these concepts in code, as well as experience in designing, training and evaluating machine learning models from scratch.
