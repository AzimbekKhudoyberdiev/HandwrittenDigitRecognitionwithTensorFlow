We implement a deep neural network consisting of convolutional and fully connected layers to classify handwritten digits of the MNIST dataset. The labeled dataset consists of 42000 images of size 28x28 = 784 pixels (one gray-scale number) including the corresponding labels from 0,..,9. The test set consists of 28000 images. Each image is normalized such that each pixel takes on values in the range [0,1]. First, we try out basic models like logistic regression, random forest and so on. After that the images are fed into the neural network, which has the following architecture:

input layer: [.,784]
layer: Conv1 -> ReLu -> MaxPool: [.,14,14,36]
layer: Conv2 -> ReLu -> MaxPool: [.,7,7,36]
layer: Conv3 -> ReLu -> MaxPool: [.,4,4,36]
layer: FC -> ReLu: [.,576]
output layer: FC -> ReLu: [.,10]
This architecture is implemented with TensorFlow. In order to prevent the network from overfitting during learning we implement dropout and data augmentation, i.e. new images are generated from the original ones via rotation, translation and zooming. Finally, we predict the digit classes for the test set and write the submission file.
Results:

The best results are achieved by using 10-fold cross validation, by stacking the neural networks on top of each other and then by training a meta model. Since each neural network is trained for 15 epochs including data augmentation which takes roughly 30 minutes on kaggle hardware, it takes in total roughly 5 hours. The final accuracy is 99.51% on the public test set. Note that we have attached saver and summary tensors to the graph, which slows down the computation.


We can also train one neural network and implement a training/validation split of 95%/5% on the labeled original images. Training on 39900 original images and including data augmentation we can achieve after 15 epochs an accuracy of roughly 99.43% on the validation set of 2100 images. Of course this can vary depending on the specific training/validation splits. It also takes roughly 30 minutes on kaggle hardware. On the public test set it can achieve an accuracy of about 99.30%. Training on all data one can actually achieve the 99.43%.

Update:

Stacking of models and training of a meta-model is now implemented.

The neural network is now implemented as a python class and the complete TensorFlow session can be saved to or restored from a file. We also implement tensor summaries, which can be visualized with TensorBoard.

Outline:

1. Libraries and settings
2.Analyze data
3.Manipulate data
4.Try out some basic models with sklearn
5. Build the neural network with TensorFlow
6. Train and validate the neural network
7. Stacking of models and training a meta-model
8.Submit the test results