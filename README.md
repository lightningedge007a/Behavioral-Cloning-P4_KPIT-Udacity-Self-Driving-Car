# Behavioral-Cloning-P4_KPIT-Udacity-Self-Driving-Car
# **BEHAVORIAL CLONING** 

[//]: # (Image References)
[image1]: ./ImagesWriteUp/nvidianet.png "this network has 27 million connections and 250 thousand parameters"
[image2]: ./ImagesWriteUp/center_2018_12_24_15_08_30_440.jpg "centre image"
[image3]: ./ImagesWriteUp/left_2018_12_24_15_08_30_440.jpg "centre image"
[image4]: ./ImagesWriteUp/right_2018_12_24_15_08_30_440.jpg "centre image"

[image5]: ./ImagesWriteUp/driving-log-output.png "data in csv file"
[image6]: ./ImagesWriteUp/steeroriginal.png "steering wheel"


## Writeup

## Pls refer to notebook Traffic_Sign_Classifier-V6

## The rubric of this project has 5 parts with (1,2,4,3,1) subparts respectively and I have written ->                  R < Part > < Subpart >       to show you all the parts in this writeup (ex: R21 is part 2 subpart 1 i.e. dataset summary in data exploration)

# R1
Project has:
- model.py file
- model.h5 file
- drive.py file
- video- run2.mp4
- writeUp
---

# R21

The model.h5 was made in Udacity workspace on keras version 2.0.8 and could be run with drive.py to obtain the video @ 60fps


# R22

The fit_generator() function is used to train the model instead of fit() which takes batch wise input (size=32) from a generator function (refer to model.py file)


==========!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!==!!!!!!!!!!!!!!!!!!!!!!!!!!!1================================
# R31

This research paper's neural net was implemented with a bit of modifications like adding dropout layers and removing a couple of layers here and there.

http://images.nvidia.com/content/tegra/automotive/images/2016/solutions/pdf/end-to-end-dl-using-px.pdf

The summary of the model I implemented is given below 

Layer (type)                 Output Shape              Param #   
=================================================================
lambda_1 (Lambda)            (None, 160, 320, 3)       0         
_________________________________________________________________
cropping2d_1 (Cropping2D)    (None, 80, 320, 3)        0         
_________________________________________________________________
conv2d_1 (Conv2D)            (None, 38, 158, 24)       1824      
_________________________________________________________________
conv2d_2 (Conv2D)            (None, 17, 77, 36)        21636     
_________________________________________________________________
conv2d_3 (Conv2D)            (None, 7, 37, 48)         43248     
_________________________________________________________________
conv2d_4 (Conv2D)            (None, 5, 35, 64)         27712     
_________________________________________________________________
conv2d_5 (Conv2D)            (None, 3, 33, 64)         36928     
_________________________________________________________________
dropout_1 (Dropout)          (None, 3, 33, 64)         0         
_________________________________________________________________
flatten_1 (Flatten)          (None, 6336)              0         
_________________________________________________________________
dense_1 (Dense)              (None, 100)               633700    
_________________________________________________________________
dense_2 (Dense)              (None, 50)                5050      
_________________________________________________________________
dense_3 (Dense)              (None, 10)                510       
_________________________________________________________________
dense_4 (Dense)              (None, 1)                 11        
=================================================================
Total params: 770,619
Trainable params: 770,619
Non-trainable params: 0
_________________________________________________________________

And here is the Nvidia's self driving car neural net 
![alt text][image1]


# R32

Train: validation split ratio is 0.8:0.2 and dropout layer is used with probability 0.5 to reduce overfitting.


# R33

Mean Square Error is used to check loss because in this situation this is the best metric which could tell how much error is in steering angle predictions and the "square" penalises the big errors. Adam optimizer is used instead of classical stochastic gradient descent as Adam performs computation for each parameter in each training step (to maintain the moving averages and variance, and calculate the scaled gradient) and retains more state for each parameter (approximately tripling the size of the model to store the average and variance for each parameter).

# R34 

Training data was collected with careful consideration with trying to keep the car in middle of the road for as long as possible. Some instances were also collected too where car drifted from sides to the middle to teach neural net on how the car should handle when it drifts off from middle of the track

# R41

After working on lenet, nvidia's architecture and trying different types of normalization the ultimate model2.py file was made. **R31** has the architecture for the model which succeeded in playing the simulation. **R33** ,**R42** and **R43** explain in detail the loss and optimization algorithms, architecture documentation and dataset augmentation respectively


# R42

The input layer normalises the images

Then the images are cropped in next layer

There are then 5 convolution layers:
	24 layers with stide 2,2 and 5x5 filter ('elu' activation)
	36 layers with stide 2,2 and 5x5 filter ('elu' activation)
	48 layers with stide 2,2 and 5x5 filter ('elu' activation)
	64 layers with 3x3 filter ('elu' activation)
	64 layers with 3x3 filter ('elu' activation)

Then there is dropout layer with prob 0.5

Then a flatten layer

Followed by 4 dense layers:
	100 units ('elu' activation)
	50 units ('elu' activation)
	10 units ('elu' activation)
	1 unit for steering angle output 
	


# R43

The dataset is biased to left, so to handle that the images are flipped and angles reversed.

This is how the dataset looks:
![alt text][image5]

This is how the bias looks like:
![alt text][image6]

 This data augmentation is somewhat equivalent to driving the track in opposite direction. A correction factor of 0.2 was taken into account to provide steering angle to left and right camera images. This multiplied the dataset I had **6** times ( 3 cameras( X 3 ) and each image flipped ( X 3 X 2) ). The images were cropped to take out the front part of car and the backgroud with trees and sky. BGR image was converted to RGB and normalized. To perform computations on such a big dataset help of a generator function was taken.

The center, left and right camera images look like this :
![alt text][image2]
![alt text][image3]
![alt text][image4]

# R51

The simulation performed seamlessly and it did encounter some sharp turns but closely managed to avoid collisions.
