# **Traffic Sign Recognition**

---

## **Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)


[image1]: ./newImages/vis1.png "Render"
[image2]: ./newImages/hist.png "Histogram"
[image10]: ./examples/visualization.jpg "Visualization"
[image20]: ./examples/grayscale.jpg "Grayscaling"
[image3]: ./examples/random_noise.jpg "Random Noise"
[image4]: ./newImages/60kph.jpg "Traffic Sign 1"
[image5]: ./newImages/7501_Gefahrstelle_01.jpg "Traffic Sign 2"
[image6]: ./newImages/Do-Not-Enter.jpg "Traffic Sign 3"
[image7]: ./newImages/intersect_row.jpg "Traffic Sign 4"
[image8]: ./newImages/roadWork.jpg "Traffic Sign 5"
[image9]: ./newImages/grayScale.png "Monochrome"

## Rubric Points
###Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one. You can submit your writeup as markdown or pdf. You can use this template as a guide for writing the report. The submission includes the project code.

You're reading it! and here is a link to my [project code](https://github.com/dzx/CarND-Traffic-Sign-Classifier-Project/blob/master/Traffic_Sign_Classifier.ipynb)

### Data Set Summary & Exploration

#### 1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

I used the numpy library to calculate summary statistics of the traffic
signs data set:

* The size of training set is 34799
* The size of the validation set is 12630
* The size of test set is 4410
* The shape of a traffic sign image is 32 x 32
* The number of unique classes/labels in the data set is 43

#### 2. Include an exploratory visualization of the dataset.

Here is an exploratory visualization of the data set. First we rendered a random example from training set just to ensure it is what we expect.

![alt text][image1]

Then there is a bar chart showing the distribution of different classes in training, validation and test sets.

![alt text][image2]

As we can see, there is an uneven distribution of different classes in training set, but at least validation and test sets roughly follow the same distribution.

### Design and Test a Model Architecture

#### 1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, and provide example images of the additional data. Then describe the characteristics of the augmented training set like number of images in the set, number of images for each class, etc.)

As a first step, I tried quick and dirty normalization by simply dividing dataset by 255 and subtracting 0.5 from result. This much was certainly necessary as LeNet seems to expect data with distribution N(0,1). I haven't done anything else at first, but I assumed that grayscaling may prove beneficial. Therefore, I implemented LeNet so that it works regardless of the last dimension of input. This way I could go between RGB and grayscale in preprocessing without having to alter the LeNet implementation.

Later on, as I experimented with pre-processing in order to boost model accuracy, I introduced more through normalization to ensure that input data actually has 0-mean and unit standard deviation. In addition, I resorted to quick grayscaling.

Here is an example of a an earlier traffic sign image and after grayscaling. Turns out Matplotlib seems to interpret resulting data as CMYK image, but conversion was still beneficial as far as LeNet was concerned.

![alt text][image9]

This is all the preprocessing I needed in order to reach the validation accuracy of 93%.


#### 2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

My final model consisted of the following layers:

| Layer         		|     Description	        					|
|:---------------------:|:---------------------------------------------:|
| Input         		| 32x32x1 RGB image   							|
| Convolution 5x5     	| 1x1 stride, valid padding, outputs 28x28x6 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 14x14x6  			  |
| Convolution 5x5	    | 1x1 stride, valid padding, outputs 10x10x16 	|
| RELU		|         									|
| Max pooling	      	| 2x2 stride,  outputs 5x5x16  			  |
| Flatten		|         	output 400							|
| Fully connected  | output  120  |
| RELU		|         									|
| Dropout  | probability .5  |
| Fully connected  | output  84  |
| RELU		|         									|
| Dropout  | probability .5  |
| Fully connected  | output  43  |




#### 3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

To train the model, I used an Adam optimizer over cross-entropy between training labels and softmax values of classifier output. Regularization has been accomplished by 2 dropout stages with same probability.

Final hyperparameters were as follows:
* Learning rate: 0.001
* Batch size: 64
* Epochs: 15
* Dropout rate: .5

#### 4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

My final model results were:
* training set accuracy of 0.993
* validation set accuracy of 0.935
* test set accuracy of 0.951

If an iterative approach was chosen:
* What was the first architecture that was tried and why was it chosen?
  * LeNet. It was chosen because project statement called for trying it out as initial architecture.
* What were some problems with the initial architecture?
  * It was really tailored for slightly different problem (handwritten digit recognition) so it was prone to overfitting.
* How was the architecture adjusted and why was it adjusted? Typical adjustments could include choosing a different model architecture, adding or taking away layers (pooling, dropout, convolution, etc), using an activation function or changing the activation function. One common justification for adjusting an architecture would be due to overfitting or underfitting. A high accuracy on the training set but low accuracy on the validation set indicates over fitting; a low accuracy on both sets indicates under fitting.
  * Upon taking hints that model might be overfitting, I started comparing training and validation error and observed that validation error remains stagnant beyond certain point as training error approaches zero, which was sign of overfitting. Therefore, dropout layers were introduced to counter the overfitting tendency.
* Which parameters were tuned? How were they adjusted and why?
  * Batch size was tuned most, as well as number of dropout layers and dropout rate. In addition, epochs count has been bumped up to 15 because 10 wasn't always enough for trained model to reach the target validation accuracy.
* What are some of the important design choices and why were they chosen? For example, why might a convolution layer work well with this problem? How might a dropout layer help with creating a successful model?
  * Knowing that traffic signs are designed to be distinguishable by color-blind people too, it was viable to reduce the input by converting images to greyscale. Both convolution and pooling layers were helpful in recognizing shapes of variable size and positioning. Finally, dropout layers combined with smaller batches improve generalization ability of the model.



### Test a Model on New Images

#### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web:

![alt text][image4] ![alt text][image5] ![alt text][image6]
![alt text][image7] ![alt text][image8]

The first image might get misclassified for another speed limit sign because they all look the same except for the one digit. The last sign got mistakenly scaled down to 30x32 which is even lower resolution than classifier input, but it still got classified right.

#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

| Image			        |     Prediction	        					|
|:---------------------:|:---------------------------------------------:|
| Speed limit (60km/h) | Speed limit (50km/h)							|
| No entry		| No entry					|
| Road work					| Road work											|
| General caution		| General caution 				|
| Right-of-way at the next intersection	| Right-of-way at the next intersection					|


The model was able to correctly guess 4 of the 5 traffic signs, which gives an accuracy of 80%. But since we are taking about only 5 samples, there is no statistically significant comparison with model accuracy on test or validation sets.

#### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The code for making predictions on my final model is located in the 20th cell of the Ipython notebook.

For the first image, the model is close to getting it right, but the right choice was second guess after all. All 5 top guesses were speed limit signs of some kind. Perhaps stacking a second model that is specialized for speed limit signs would be beneficial for cases like this.

| Probability         	|     Prediction	        					|
|:---------------------:|:---------------------------------------------:|
| .497         			| Speed limit (50km/h)							|
| .468     				| Speed limit (60km/h) 										|
| .02955					| Speed limit (30km/h)						|
| .00390      			| Speed limit (80km/h)		 				|
| .000820446		    | End of speed limit (80km/h)							|


For the second image model is almost certain that it is a 'no entry' sign which is correct. Confidence levels for remaining 4 guesses are infinitesimally small.

| Probability         	|     Prediction	        					|
|:---------------------:|:---------------------------------------------:|
| 1     			| No entry					|
| 8.78179582e-13  		| Speed limit (20km/h)										|
| 7.98556750e-13  		| Stop										|
| 7.11973205e-14		| Turn left ahead						|
| 5.14469821e-14 		| Keep right						|

For the third image, model is almost certain that it is a 'Road work' sign which is correct. Confidence levels for remaining 4 guesses are infinitesimally small.

| Probability         	|     Prediction	        					|
|:---------------------:|:---------------------------------------------:|
| .99999		| Road work			|
| 4.50136184e-07		| Dangerous curve to the right	|
| 4.27288072e-09	| Keep right	|
| 5.41977123e-11	| No passing for vehicles over 3.5 metric tons	|
| 1.12939246e-13	| General caution	|

For the fourth image, model is almost certain that it is a 'General caution' sign which is correct. Confidence levels for remaining 4 guesses are infinitesimally small.

| Probability         	|     Prediction	        					|
|:---------------------:|:---------------------------------------------:|
| 1	| General caution	|
| 1.05231162e-11	| Traffic signals	|
| 1.88609196e-21	| Right-of-way at the next intersection	|
| 1.30556596e-21	| Pedestrians	|
| 1.20213670e-31	| Go straight or left	|


For the fifth image, model is almost certain that it is a 'Right-of-way at the next intersection' sign which is correct. Confidence levels for remaining 4 guesses are infinitesimally small.

| Probability         	|     Prediction	        					|
|:---------------------:|:---------------------------------------------:|
| 1	| Right-of-way at the next intersection	|
| 1.86920229e-10	| Beware of ice/snow	|
| 9.01936783e-13	| Double curve	|
| 2.88244417e-13	| Pedestrians	|
| 1.20213670e-31	| Speed limit (20km/h)	|

### (Optional) Visualizing the Neural Network (See Step 4 of the Ipython notebook for more details)
#### 1. Discuss the visual output of your trained network's feature maps. What characteristics did the neural network use to make classifications?

Since project is already past due, not doing this.
