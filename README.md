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

[image1]: ./images/left.jpg "image from left camera"
[image2]: ./images/center.jpg "image from center camera"
[image3]: ./images/right.jpg "image from right camera"
[image4]: ./images/original.jpg "original image"
[image5]: ./images/flipped.jpg "flipped image"
[image6]: ./images/figure_1.png ""

---
### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

First of all, my model will preprocess the data by normalizing and cropping.

After preprocessing, my model has 5 convolution2D layers followed by 1 flatten layer followed by 5 fully-connected layers(model.py lines 64-78) 

The model includes RELU layers to introduce nonlinearity (code lines 64-66). 

#### 2. Attempts to reduce overfitting in the model
The model contains dropout layers in order to reduce overfitting (model.py lines 71, 73, 75 and 77). 

The model was trained and validated on different data sets to ensure that the model was not overfitting (code line 10-16). The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

#### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (model.py line 80).

#### 4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used more than 15 minutes of data(51,720 images/about 800 MBs of data).

I use all of the three cameras(left, right and center) and assumes the steering_correction factor is 0.12 by trial and error.

I also flipped the image to double the size of my data by using np.fliplr() and multiply the steering angle by -1  

For details about how I created the training data, see the next section. 

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

The model architecture was the one NVIDIA published in their self-driving behavior cloning paper, which is introduced in the lesson.

One difference between mine and the one in the lesson is that I introduced dropout layer, after fully-connected layers except for the last FC layer, to avoid overfitting.

I also found that sometimes my behavior cloning output left the center line a little bit, that is because in the 15 minutes recording, I've deviated from the center line several times after making a sharp turn(because of my sloppy maneuver). Since this model is only doing behavior cloning, we can not avoid this if traning data is not perfect.

#### 2. Final Model Architecture

The final model architecture (model.py lines 60-86) consisted of a convolution neural network with 5 convolution layers, a flatten layer, 5 fully-connected layers interweaving with 4 dropout layers

Here is the architecture of my final model:

| Layer         		|     Description	        									| 
|:---------------------:|:-------------------------------------------------------------:| 
| Input         		| 160x320x3 Color image 	  									| 
| Normalization     	| realized with "lambda x:x/255.0-0.5"		 					|
| Cropping				| cropped the top 70 rows and bottom 25 rows of pixels			|
| Convolution 5x5  		| nb_filter = 24, strides = 2x2, use RELU as activation method	|
| Convolution 5x5  		| nb_filter = 36, strides = 2x2, use RELU as activation method	|
| Convolution 5x5  		| nb_filter = 48, strides = 2x2, use RELU as activation method	|
| Convolution 3x3  		| nb_filter = 64, use RELU as activation method					|
| Convolution 3x3  		| nb_filter = 64, use RELU as activation method					|
| Flatten				| flatten the output from last layer 							|
| Fully Connected Layer	| output dimension = 100										|
| Dropout Layer			| drop out rate = 50%											|
| Fully Connected Layer	| output dimension = 50											|
| Dropout Layer			| drop out rate = 50%											|
| Fully Connected Layer	| output dimension = 10											|
| Dropout Layer			| drop out rate = 50%											|
| Fully Connected Layer	| output dimension = 5											|
| Dropout Layer			| drop out rate = 50%											|
| Fully Connected Layer	| output dimension = 1											|

#### 3. Creation of the Training Set & Training Process

To capture good driving behavior, I recorded 15 minutes of data using center lane driving. Here are 3 example images for the 3 cameras(left, center and right) for one identical moment in the recording:

![alt text][image1]
![alt text][image2]
![alt text][image3]

To augment the data sat, I also flipped images using fliplr() in python's numpy module and multiply the original steering angle by -1. For example, here is an image that has then been flipped(left: original image, right: flipped image):

![alt text][image4]
![alt text][image5]

After the collection process, I had 11634 data points. I then preprocessed this data by batch size of 32 and using generator method to avoid exceeding memory size limit.

I finally randomly shuffled the data set and put 20% of the data into a validation set. 

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was 5 as evidenced by the decreasing trend of validation loss shown in visualization.
![alt text][image6]
I used an adam optimizer so that manually training the learning rate wasn't necessary.

However, as I mentioned in the previous section, my behavior cloning output left the center line a little bit, that is because I've deviated from the center line several times in the 15 minutes recording. Since this model is only doing behavior cloning, this simple approach is severely vulnerable by imperfect data.

Furthermore, the speed for training was around 30 Mph during the 15 minutes whhich is different from testing, however in this project we don't control car speed as one variable, we cannot really control it but this is definitely a major factor we need to consider if we want to further expand this project.