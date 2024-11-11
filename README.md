# Convolutional Autoencoder
This read me is taken from the second assignment of the Advance Concepts in Machine Learning(ACML) course of the Artificial Intelligence master's programme of the Maastricht University. 

An autoencoder (AE) is a neural network that is trained to copy its input to its output. It has one or more hidden layers h that each describe a code used to represent the input. The network may be viewed as consisting of two parts: an encoder h = f(x) that produces the code and a decoder that produces a reconstruction of the data r = g(h).
The encoder is tasked with finding a (usually) reduced representation of the data, extracting the most prominent features of the original data and representing the input in terms of these features in a way the Decoder can understand.
The Decoder learns to read these codes and regenerate the data from them. The entire AE aims to minimize a loss-function while reconstructing. In their simplest form encoder and decoder are fully-connected feedforward neural networks. When the inputs are images, it makes sense to use convolutional neural networks (CNNs) instead, obtaining a convolutional autoencoder (CAE).
A CAE uses the convolutional filters to extract features. CAEs learn to encode the input as a combination of autonomously learned signals and then to reconstruct the input from this encoding. CAEs learn in what can be seen as an unsupervised learning setup, since they don’t require labels and instead aim to reconstruct the input. The output is evaluated by comparing the reconstructed image by the original one, using a Mean Squared Error (MSE) cost function.

# Data
In this lab we will be using the CIFAR-10 dataset that you can find in the link below:
• CIFAR-10 dataset (https://www.cs.toronto.edu/ ̃kriz/cifar.html).
The CIFAR-10 dataset consists of 60 000 thousand color images of dimensions 32 × 32. Images can be of any of 10 categories, and each image may only be of one such category, although for this lab the category of the images is largely irrelevant. Here are some examples of what to expect: 

![Sample Image](images/Cifar_image.png)

Links are provided to download the dataset in a format already prepared for python or Matlab. However, since this dataset is so common, there is likely a binding in your preferred machine learning toolkit that will download the data for you in a format ready to be used. For instance:

- **Keras**: [Keras CIFAR-10 Dataset](https://keras.io/api/datasets/cifar10/)
- **Matlab**: Look up the `helperCIFAR10Data` function.
- **PyTorch**: Look up the `torchvision.datasets.CIFAR10` class.

# Reconstruction 
We use the library:
**Keras**: [Keras Documentation for CNNs(https://keras.io/guides/keras_applications/), to construct a convolutional autoencoder that takes an image from the CIFAR-10 dataset as input, produces a representation in latent space, and attempts to reconstruct the original image as precisely as possible. The only data the autoencoder requires is the colored image, which will act as both the input and the output to compare against. 

The network consists of a total of 9 layers. The input images (of size 32 × 32 × 3) are fed into a convolutional layer with filter size 3 × 3 and 8 channels (or dimensions). This is followed by a max pooling layer which downsamples the image with a 2 × 2 filter, effectively reducing the width and height of the image by a factor of 2, thereby reducing the total image area by a factor of 4. Then, another layer of filter size 3 × 3 and 12 channels follows.
