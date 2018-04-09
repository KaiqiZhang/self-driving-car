# **Traffic Sign Recognition**

## Writeup

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./visualize_dataset.jpg "Visualization"
[image2]: ./visualize_norm.jpg "Normalization"
[image3]: ./examples/random_noise.jpg "Random Noise"
[image5]: ./Traffic_Signs/00317.jpg "Traffic Sign 2"
[image4]: ./Traffic_Signs/00579.jpg "Traffic Sign 1"
[image8]: ./Traffic_Signs/08456.jpg "Traffic Sign 5"
[image7]: ./Traffic_Signs/09714.jpg "Traffic Sign 4"
[image6]: ./Traffic_Signs/11987.jpg "Traffic Sign 3"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.

---
### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one. You can submit your writeup as markdown or pdf. You can use this template as a guide for writing the report. The submission includes the project code.

You're reading it! and here is a link to my [project code](https://github.com/KaiqiZhang/self-driving-car/blob/master/CarND-Traffic-Sign-Classifier-Project/Traffic_Sign_Classifier.ipynb)

### Data Set Summary & Exploration

#### 1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

I used the pandas library to calculate summary statistics of the traffic
signs data set:

* The size of training set is 34799
* The size of the validation set is 4410
* The size of test set is 12630
* The shape of a traffic sign image is (32, 32, 3)
* The number of unique classes/labels in the data set is 43

#### 2. Include an exploratory visualization of the dataset.

Here is an exploratory visualization of the training data set. It is a bar chart showing how the data distribute across the traffic sign classes. The x axis represents the 43 classes of traffic signs. The y axis represents the number of images in each class.

![alt text][image1]

### Design and Test a Model Architecture

#### 1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, and provide example images of the additional data. Then describe the characteristics of the augmented training set like number of images in the set, number of images for each class, etc.)

I just do normalization to the dataset for preprocessing. First, I compute the means of each channel from the training set. Second, I substract the means from training, validation, and testing dataset. Finally, I divide all the data by 256.0 so that they won't saturate.

![alt text][image2]

I didn't convert the images to grayscale because I think neural network can get some useful information from the color channels.

However, I believe some data augmentation can make the accuracy get higher, for example, randomly flipping, scaling, or rotating the images.

#### 2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

My final model consisted of the following layers:

| Layer         		|     Description	        					|
|:---------------------:|:---------------------------------------------:|
| Input         		| 32x32x3 RGB image   							|
| Convolution 3x3     	| 1x1 stride, valid padding, outputs 28*28*20 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 14x14x20 				|
| Convolution 3x3	    | 1x1 stride, valid padding, outputs 10*10*50 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 5x5x50 				|
| Fully connected		| outputs 500        									|
| Dropout		| keep probability 50%        									|
| Fully connected		| outputs 43        									|
| Logits				| logits        									|

#### 3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

To train the model, I used an Adam optimizer. The batch size is 128 and number of epochs is 10. The initial learning rate is 0.001, then the Adam optimizer will automatically tune the learning rate.

#### 4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

My final model results were:
* training set accuracy of 1.0
* validation set accuracy of 0.953
* test set accuracy of 0.947

If an iterative approach was chosen:
* What was the first architecture that was tried and why was it chosen?
* What were some problems with the initial architecture?
* How was the architecture adjusted and why was it adjusted? Typical adjustments could include choosing a different model architecture, adding or taking away layers (pooling, dropout, convolution, etc), using an activation function or changing the activation function. One common justification for adjusting an architecture would be due to overfitting or underfitting. A high accuracy on the training set but low accuracy on the validation set indicates over fitting; a low accuracy on both sets indicates under fitting.
* Which parameters were tuned? How were they adjusted and why?
* What are some of the important design choices and why were they chosen? For example, why might a convolution layer work well with this problem? How might a dropout layer help with creating a successful model?

If a well known architecture was chosen:
* What architecture was chosen?
* Why did you believe it would be relevant to the traffic sign application?
* How does the final model's accuracy on the training, validation and test set provide evidence that the model is working well?

I choose to use the hyperparameters almost the same as LeNet 5, except the input channel is 3 rather than 1 in MNIST. Because convolution network has been proved to very effective to image recognition.

The model achieve a good accuracy at the first attempt. However, I found the convergence speed was slow, and the final validation accuracy is only a bit higher than 0.93.

Then I decided to add dropout layer because it can effectively prevent model from overfitting. The result is very good. Model convergence much faster, and the final validation accuracy achieved 0.96.

### Test a Model on New Images

#### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web:

![alt text][image4] ![alt text][image5] ![alt text][image6]
![alt text][image7] ![alt text][image8]

The first image might be difficult to classify because the number is the red circle is difficult to discriminate.

The second image might be difficult to classify because many traffic signs are round and with blue background as well as contains arrows in it.

The third image might be difficult to classify because if can be confused with the speed limit signs.

The fourth image might be difficult to classify because many traffic signs are round and with blue background as well as contains arrows in it.

The fifth image might be difficult to classify because it's very dark and vague. Even human is difficult to discriminate the car symbol in the center of the circle.

#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

| Image			        |     Prediction	        					|
|:---------------------:|:---------------------------------------------:|
| Speed limit (20km/h) | Speed limit (20km/h)	|
| Roundabout mandatory     			| Roundabout mandatory 										|
| No passing					| No passing											|
| Keep right	      		| Keep right					 				|
| End of no passing			| End of no passing      							|


The model was able to correctly guess 5 of the 5 traffic signs, which gives an accuracy of 100%. This compares favorably to the accuracy on the test set of 0.947.

#### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The code for making predictions on my final model is located in the 11th cell of the Ipython notebook.

For the first image, the model is pretty sure that this is a Speed limit (20km/h) (probability of 0.9999), and the image does contain a Speed limit (20km/h). The most high score predictions are all Speed limit. The top five soft max probabilities were

| Probability         	|     Prediction	        					|
|:---------------------:|:---------------------------------------------:|
| .9999         			| Speed limit (20km/h)						|
| 1.01e-4     				| Speed limit (30km/h)				|
| 1.57e-5					    | Speed limit (70km/h)					|
| 6.81e-6	      			| Speed limit (120km/h)				|
| 1.00e-10				    | No passing					|

For the second image, the model is pretty sure that this is a Roundabout mandatory (probability of 1.0000), and the image does contain a Roundabout mandatory. The top five soft max probabilities were

| Probability         	|     Prediction	        					|
|:---------------------:|:---------------------------------------------:|
| 1.0000         			| Roundabout mandatory   									|
| 3.10e-8     				| Go straight or left 										|
| 2.13e-8					    | Keep left											|
| 6.63e-9	      			| Turn right ahead					 				|
| 3.01e-9				    | Beware of ice/snow      							|

For the third image, the model is pretty sure that this is a No passing (probability of 1.0000), and the image does contain a No passing. The top five soft max probabilities were

| Probability         	|     Prediction	        					|
|:---------------------:|:---------------------------------------------:|
| 1.0000        			| No passing   									|
| 3.32e-6     				| No passing for vehicles over 3.5 metric tons |
| 4.86e-7					    | Speed limit (100km/h)						|
| 1.35e-8	      			| Vehicles over 3.5 metric tons prohibited			|
| 1.16e-9 				    | No entry						|

For the fourth image, the model is pretty sure that this is a Keep right (probability of 1.0000), and the image does contain a Keep right. The top five soft max probabilities were

| Probability         	|     Prediction	        					|
|:---------------------:|:---------------------------------------------:|
| 1.0000         			| Keep right   									|
| 1.16e-12    				| Turn left ahead 										|
| 1.27e-13					  | Roundabout mandatory											|
| 5.82e-17      			| End of no passing					 				|
| 2.05e-18				    | Beware of ice/snow      							|

For the fifth image, the model is pretty sure that this is a End of no passing (probability of 1.0000), and the image does contain a End of no passing. The top five soft max probabilities were

| Probability         	|     Prediction	        					|
|:---------------------:|:---------------------------------------------:|
| 1.0000         			| End of no passing   									|
| 3.75e-7     				| End of no passing by vehicles over 3.5 metric tons |
| 1.96e-10				    | No passing		|
| 6.79e-14      			| Priority road		|
| 2.83e-14				    | Dangerous curve to the right	|

### (Optional) Visualizing the Neural Network (See Step 4 of the Ipython notebook for more details)
#### 1. Discuss the visual output of your trained network's feature maps. What characteristics did the neural network use to make classifications?
