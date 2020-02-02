## Emotion Recognition using Pytorch

Implementation for emotion detection using pytorch 

### Objective
To build a model that can detect and localize specific objects in images.

The dataset is built by first using the xml files to get the filename, height, width, bounding box for each image (including those with multiple objects in an image). 

We then read an image and perform the required transformations and return and image, label, and bounding box. 

Howver, in order to boost out dataset we use chromedriver to first download images with similar features. 

Finding the faces : Before we train an emotion detector, we first must be able to locate the faces within each image. Since the goal is to classify emotions from facial expressions, it makes sense to only keep the faces we find and throw away all other (mostly irrelevant) features. For this project, we will use the facenet-pytorch library which provides a multi-task CNN pre-trained on the VGGFace2 and CASIA-Webface datasets.There after, we proceed to use the principle of transfer learning to get a model (Resnet-50) trained on imagenet dataset. 

Since we don’t have a large dataset, we should avoid training our classifier from scratch. As is common in most computer vision transfer learning tasks, we will go ahead and fine-tune a model that is pre-trained on ImageNet, a dataset with millions of images across a thousand diverse labels.

The goal here is to make it easier for our model to train by giving its parameters a good starting point. It can leverage the existing feature extractors learned from ImageNet and tweak them to work on this new task.

A caveat to ImageNet pre-trained models is that they expect their input to be an RGB image. We could side-step this issue by transforming our images into three-channel grayscale images. Though this would work, it is not optimal since we are extrapolating the model to a new feature space (ImageNet contains no grayscale images).

Instead, let’s swap out the first (three-channel) convolutional layer in the model with a randomly initialized one-channel convolutional layer. We will be using Resnet-50 as our model.Here, an RGB image is fed as input to a pretrained Resnet-50 model. The image is also converted into grayscale and fed as input to the modified “Gray Resnet-50”, where the first convolution layer now accepts single-channel input. The two output embeddings are then passed through an L2 loss function. Though there are a lot of computations going on, the only parameters being updated in the backpropagation step is the first, single-channel convolutional layer in the “Gray Resnet” model.

Now that we have our base model, let’s apply transfer learning to classify emotions! This will involve a two-stage fine-tuning of our pre-trained “Gray Resnet” model.

The first stage involves fine-tuning on the FER dataset. This will serve as an intermediate step before the model is trained on the “wild” data. The hope is that the model can learn to identify simple features from the FER dataset due to the coarse resolution of the images (48 x 48).

The second stage takes the model from the first stage and then further fine-tunes it on the “wild” data. Here, the model will extend the previous representation to learn from richer features present in the “wild” data.

The reason why the entire model was not allowed to train during the second stage is because there are fewer examples in the “wild” dataset. Given its large capacity, the model would be more prone to overfitting if all of its parameters were allowed to train in that stage.

We then can apply the model to any test image and obtain the emotions. 





