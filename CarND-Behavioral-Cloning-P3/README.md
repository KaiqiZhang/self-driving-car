# **Behavioral Cloning**

---

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./docs/cnn-architecture.png "Nvidia Model Architecture"
[image2]: ./docs/viz_dataset.png "Dataset Visualization"
[image3]: ./docs/left.jpg "Left Camera"
[image4]: ./docs/center.jpg "Center Camera"
[image5]: ./docs/right.jpg "Right Camera"
[image6]: ./docs/demo.gif "Demo Gif"

---

## 1. Project Overview

My project includes the following files:
* `model.py` containing the script to create and train the model
* `drive.py` for driving the car in autonomous mode
* `model.h5` containing a trained convolution neural network
* `writeup.md` summarizing the results

**Train model:**
```
python model.py
```
**Test model in simulator**
```
python drive.py model.h5
```

## 2. Model Architecture
After reading the course materials. I choose to use the network parameter from the [Nvidia Papaer](https://devblogs.nvidia.com/deep-learning-self-driving-cars/) mentioned in the course material. It comes out to be quite suitable for this task. The model size is small (~4MB), but the performance is pretty good. It contains 5 convolutional layers, 3 fully-connected layers and a single neuron to compute the steering angle. Connected to the input is a normalization layer (lambda) and a cropping layer. Additionally, I use ReLU for the activations.

The model architecture is shown below:

![alt text][image1]

## 3. Data augmentation
### Visualizing the dataset
I use the dataset downloaded from the Udacity course materials. The distribution of steering angles is shown below. As we can see, most of them is the case when car is in the center of the track. For a proper training, we should make the model learn how to deal with different situations.

![alt text][image2]

To avoid this problem taken by dataset, I used the following augmentation techniques in Keras data generator to generate new images.

### Crop the images
The top portion of the image captures trees, hills and sky. The bottom portion of the image captures the hood of the car. To avoid these disturbs, I cropped:

* 70 row pixels from the top of the image
* 25 row pixels from the bottom of the image
* 0 column of pixels from the left of the image
* 0 column of pixels from the right of the image

### Randomly choose camera
Each frame of the dataset contains pictures from 3 cameras: center, left, and right. We can. In the data generator, I made the code randomly choose a image from center, left or right camera. Of course, we should shift the steering angle for the corresponding image. For the left image, steering angle is adjusted by +0.2. For the right image, steering angle is adjusted by -0.2.

Left | Center | Right
-----|--------|-------
![alt text][image3] | ![alt text][image4] | ![alt text][image5]

### Randomly flipping
Just by flipping the image, we can get a new input data. (Of course, we should multiple the target steering angle by -1.0) This is a effective technique to deal with left turn bias. In the data generator, I made the code randomly choose to flip a image or not.

## 4. Training and Fine-tuning
To avoid loading to many images to the memory, I use two generators to generator the training and testing batches. I used `fit_generator` API of Keras to conduct batch training and testing. The batch size is set to 32.

I use the Adam optimizer to train the model. The default parameter comes out to be very good for optimization. Totally 5 epoches are trained. I use a GTX 1080Ti GPU and the training finished in 3 minutes.

In each batch, the generator first randomly choose a picture from center, left or right camera, and then randomly choose to flip the picture or not. Thus generate augmented data.

### Prevent overfitting
To prevent overfitting, I add a dropout layer between the last convolutional layer and the first fully-connected layer. The dropout probability is set to 0.5. The final training and validation loss come out to be almost equal.

### Split training and testing data
The model was trained and validated on different data sets to ensure that the model was not overfitting. 20% of the dataset is put into the validation data. The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

### Model parameter tuning
I tuned the shifting steering angle for the images from left and right camera. At first, I set it to 0.25, the car seems a litter oscillating between the sides. Then I tune it to 0.2. The validation loss decreased a lot, and the car seems running more smoothly.

## 5. Test Results
With the model, the car can drive smoothly on the track without bumping into the side ways. Just like a human! See demo video below:

![alt text][image6]

[YouTube Link](https://youtu.be/iBmzKmPdZIo)
