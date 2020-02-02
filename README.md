## Emotion Recognition using Pytorch

Implementation for emotion detection using pytorch 

### Objective
To build a model that can detect and localize specific objects in images.

The dataset is built by first using the xml files to get the filename, height, width, bounding box for each image (including those with multiple objects in an image). 

We then read an image and perform the required transformations and return and image, label, and bounding box. 

There after we proceed to use the principle of transfer learning to get a model (Resnet-50) trained on imagenet dataset. 
Here, an RGB image is fed as input to a pretrained Resnet-50 model. The image is also converted into grayscale and fed as input to the modified “Gray Resnet-50”, where the first convolution layer now accepts single-channel input. The two output embeddings are then passed through an L2 loss function. Though there are a lot of computations going on, the only parameters being updated in the backpropagation step is the first, single-channel convolutional layer in the “Gray Resnet” model. 




