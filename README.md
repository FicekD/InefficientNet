## InefficientNet
Numpy deep learning implementation. Trying to achieve similar style to Keras.

Goal is to get sequential convnets working, maybe something like Keras' functional API, maybe convolution implementation in C.

### What's working
- Training loop - forward and backward propagation
- Dense, Flatten and Dropout layers
- Categorical crossentropy loss
- ReLU, Leaky ReLU, Sigmoid activations
- Gradient descent optimizer

### What's not working
- Softmax activation
- Adam optimizer
- Convolutional layers stride other than 1
- Backpropagation of Convolutional, Maxpool, Dropout layers

Currently testing on reduced MNIST to make sure everythong works together as intended. Unit testesting forward propagations.