#Generate multiple data out of one single picture by shrinking, zooming, cropping etc.
from keras.preprocessing.image import ImageDataGenerator
#To model neural network
from keras.models import Sequential
#Conv2D: Convolution neural network to extract features of the picture
#Maxpooling2D: Reduces size of data/picture
from keras.layers import Conv2D, MaxPooling2D
#Activation: Will tell neural network when to activate to 1 or -1 or 0
#Flatter: It is a function that converts 2D image to 1D image and put it into the neural network
#Dense: This is used to create a hidden layer or output layer. Dense basically helps in creating layers
from keras.layers import Activation, Dropout, Flatten, Dense
#This function tells us which channel comes first. Like (3layer, width,height)
from keras import backend as K
#To manipulate arrays
import numpy as np
#Preprocessing import image-to import images directly from the directory and processes them
from keras.preprocessing import image


img_width, img_height = 150, 150

train_data_dir = 'Dataset/Train'
validation_data_dir = 'Dataset/Validation'
nb_train_samples = 324
nb_validation_samples = 12
epochs = 50
batch_size = 20

#Set all images into a format (3, width, height) checks if image in right format or not
if K.image_data_format() == 'channels_first' :
    input_shape = (3, img_width, img_height)
else :
    input_shape = (img_width, img_height, 3)

#Collecting data from one image by rescaling it, shearing it, zooming it and horizontally flipping it.
train_datagen = ImageDataGenerator(
        rescale = 1. / 255,
        shear_range = 0.2,
        zoom_range = 0.2,
        horizontal_flip = True)


test_datagen = ImageDataGenerator(rescale = 1. / 255)

#To get all images from directory and process it
train_generator = train_datagen.flow_from_directory(
        train_data_dir,
        target_size = (img_width, img_height),
        batch_size = batch_size,
        class_mode = 'binary')

validation_generator = test_datagen.flow_from_directory(
        validation_data_dir,
        target_size = (img_width, img_height),
        batch_size = batch_size,
        class_mode = 'binary')

#Adding CNN, 32 means extract 32 features from the image, size of searching matrix: 3*3 pixels
model = Sequential()
model.add(Conv2D(32, (3, 3), input_shape=input_shape))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size = (2, 2)))

model.summary()

model.add(Conv2D(32, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

#Extracting 64 features from it
model.add(Conv2D(64, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

#Flattening image-2D to 1D
#Need hidden layers that activate with given data and then gives output as 64 cause 64 features as input
model.add(Flatten())
model.add(Dense(64))
model.add(Activation('relu'))
#For this output layer function is sigmoid
model.add(Dropout(0.5))
model.add(Dense(1))
model.add(Activation('sigmoid'))

#Everything done till now is displayed
model.summary()

#To compile with loss
model.compile(loss='binary_crossentropy', optimizer = 'rmsprop', metrics = ['accuracy'])

#Feeding data to model
model.fit_generator (
        train_generator,
        steps_per_epoch = nb_train_samples // batch_size,
        epochs = epochs,
        validation_data = validation_generator,
        validation_steps = nb_validation_samples // batch_size)

#model.save_weights('model.h5')
#Saving model into a .h5 file
model.save('model.h5')

img_pred = image.load_img('nikiopen.jpg', target_size = (150, 150))
img_pred = image.img_to_array(img_pred)
img_pred = np.expand_dims(img_pred, axis = 0)

#If the result is 1 then open will be predicted or else close
rslt = model.predict(img_pred)
print (rslt)
if rslt[0][0] == 1 :
    prediction = "open"
else :
    prediction = "closed"

#Prinitng the prediction as open or close
print (prediction)
