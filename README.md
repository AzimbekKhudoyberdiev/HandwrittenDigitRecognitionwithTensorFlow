# HandwrittenDigitRecognitionUsingTensorFlow
 Handwritten Digit Recognition Using TensorFlow
We implement a deep neural network consisting of convolutional and fully connected layers to classify handwritten digits of the MNIST dataset. The labeled dataset consists of 42000 images of size 28x28 = 784 pixels (one gray-scale number) including the corresponding labels from 0,..,9. 

The test set consists of 28000 images. Each image is normalized such that each pixel takes on values in the range [0,1]. First, we try out basic models like logistic regression, random forest and so on. After that the images are fed into the neural network, which has the following architecture:

input layer: [.,784]
layer: Conv1 -> ReLu -> MaxPool: [.,14,14,36]
layer: Conv2 -> ReLu -> MaxPool: [.,7,7,36]
layer: Conv3 -> ReLu -> MaxPool: [.,4,4,36]
layer: FC -> ReLu: [.,576]
output layer: FC -> ReLu: [.,10]

This architecture is implemented with TensorFlow. In order to prevent the network from overfitting during learning we implement dropout and data augmentation, i.e. new images are generated from the original ones via rotation, translation and zooming. Finally, we predict the digit classes for the test set and write the submission file.
Outline of the Project includes 8 main steps:
1. Libraries;
2. Data Analyzing;
3. Data Manipulation;
4. Basic models with Sklearn;
5. Build the Network with TensorFlow;
6. Training and Validation of Neural Network;
7. Stacking of Models and Training a meta-model;
8. Validation of Test Results; 

