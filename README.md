## InefficientNet
Numpy deep learning implementation. Trying to achieve similar style to Keras.

Goal is to get sequential convnets working, maybe something like Keras' functional API, maybe convolution implementation in C.

### What's working
- Training loop - forward and backward propagation for fully connected layers
- Categorical crossentropy and MSE loss
- ReLU, Leaky ReLU, Tanh, Sigmoid and Softmax activations
- Gradient descent optimizer

### What's not working
- Adam optimizer
- Backpropagation of Convolutional, Maxpool and Dropout layers

Currently testing on reduced MNIST to make sure everything works together as intended. Unit testesting forward propagations.