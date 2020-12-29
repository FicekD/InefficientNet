## InefficientNet
Numpy deep learning implementation. Trying to achieve similar style to Keras.

Goal is to get sequential convnets working, maybe something like Keras' functional API, maybe convolution implementation in C.

### What's working
- Training loop - forward and backward propagation for fully connected networks
- Categorical crossentropy and MSE losses
- ReLU, Leaky ReLU, Tanh, Sigmoid and Softmax activations
- Gradient descent and Adam optimizers

### What's not working
- Backpropagation of Convolutional, Maxpool and Dropout layers

Currently testing on reduced MNIST to make sure everything works together as intended. Unit testesting forward propagations.