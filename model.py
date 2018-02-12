import csv
import cv2
import numpy as np
from keras.models import Sequential
from keras.layers import Flatten, Dense,Lambda,Cropping2D
from keras.layers.pooling import MaxPooling2D
from keras.layers.convolutional import Convolution2D

from sklearn.model_selection import train_test_split
import sklearn

def generator(samples, batch_size=32):
	num_samples = len(samples)
	while 1: # Loop forever so the generator never terminates
		#shuffle(samples)
		for offset in range(0, num_samples, batch_size):
			batch_samples = samples[offset:offset+batch_size]

			images = []
			angles = []
			correction = 0.12 # this is a parameter to tune
			for batch_sample in batch_samples:
				center_image = cv2.imread(batch_sample[0])
				left_image = cv2.imread(batch_sample[1])
				right_image = cv2.imread(batch_sample[2])
				center_angle = float(batch_sample[3])
				left_angle = center_angle + correction
				right_angle = center_angle - correction
				images.extend([center_image,left_image,right_image])
				angles.extend([center_angle,left_angle,right_angle])
				images.extend([np.fliplr(center_image),np.fliplr(left_image),np.fliplr(right_image)])
				angles.extend([-center_angle,-left_angle,-right_angle])
				
			# trim image to only see section with road
			X_train = np.array(images)
			y_train = np.array(angles)
			yield sklearn.utils.shuffle(X_train, y_train)
			
lines =[]
with open('data/driving_log.csv') as csv_file:
	reader = csv.reader(csv_file)
	for line in reader:
		lines.append(line)


train_samples, validation_samples = train_test_split(lines, test_size=0.2)

images = []
image_names = []
measurements = []

"""
for line in lines:
	steering_center = float(line[3])

	# create adjusted steering measurements for the side camera images
	correction = 0.12 # this is a parameter to tune
	steering_left = steering_center + correction
	steering_right = steering_center - correction

	# read in images from center, left and right cameras
	#img_center = cv2.imread(line[0])
	#img_left = cv2.imread(line[1])
	#img_right = cv2.imread(line[2])

	# add images and angles to data set
	image_names.extend(line[0:3])
	measurements.extend([steering_center, steering_left, steering_right])

	images.extend([img_center, img_left, img_right])
	measurements.extend([steering_center, steering_left, steering_right])
	images.extend([np.fliplr(img_center), np.fliplr(img_left), np.fliplr(img_right)])
	measurements.extend([-steering_center, -steering_left, -steering_right])
"""


train_generator = generator(train_samples, batch_size=32)
validation_generator = generator(validation_samples, batch_size=32)

X_train = np.array(images)
y_train = np.array(measurements)

my_cropping=((70,25), (0,0))
model = Sequential()
model.add(Lambda(lambda x:x/255.0-0.5,input_shape=(160,320,3)))
model.add(Cropping2D(cropping=my_cropping, input_shape=(3,160,320)))

model.add(Convolution2D(24,5,5,subsample=(2,2),activation='relu'))
model.add(Convolution2D(36,5,5,subsample=(2,2),activation='relu'))
model.add(Convolution2D(48,5,5,subsample=(2,2),activation='relu'))
model.add(Convolution2D(64,3,3,activation='relu'))
model.add(Convolution2D(64,3,3,activation='relu'))
model.add(Flatten())
model.add(Dense(100))
model.add(Dense(50))
model.add(Dense(10))
model.add(Dense(5))
model.add(Dense(1))

model.compile(loss='mse',optimizer='adam')
#model.fit(X_train,y_train,validation_split=0.2,shuffle=True,nb_epoch=4)

model.fit_generator(train_generator, 
	samples_per_epoch=len(train_samples), 
	validation_data=validation_generator,
	nb_val_samples=len(validation_samples), 
	nb_epoch=3)

model.save('model.h5')