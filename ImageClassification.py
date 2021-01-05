#Tries to generate multiple data out of one isngle picture by shrinking,zooming, cropping etc
from keras.preprocessing.image import ImageDataGenerator 
#To model neural network
from keras.models import Sequential    
#Conv2D to extract features of picture, MaxPooling2D to reduce size of picture
from keras.layers import Conv2D, MaxPooling2D     
from keras.layers import Activation, Dropout, Flatten, Dense   
from keras import backend as K
#Used to manipulate arrays
import numpy as np   
#Preprocessing import image to import them directly from dictionary and processes them
from keras.preprocessing import image  


img_width, img_height = 150, 150

train_data_dir = 'Dataset/Train'
validation_data_dir = 'Dataset/Validation'
nb_train_samples = 324
nb_validation_samples = 12
epochs = 50
batch_size = 20

#To set all images into a format(3,width,height) and checks if image in right format or not
if K.image_data_format() == 'channels_first' :
    input_shape = (3, img_width, img_height)
else :
    input_shape = (img_width, img_height, 3)
#Collects data from one image by recalling, shearing, zooming, horizontally flipping it
train_datagen = ImageDataGenerator(
        rescale = 1. / 255,
        shear_range = 0.2,
        zoom_range = 0.2,
        horizontal_flip = True)

#Testing data to be natural so we are rescaling it
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

#Model is the name of neural network 
model = Sequential()
#Adding CNN 32 means extract 32 features from image, size of searching matrix is 3*3 pixels
model.add(Conv2D(32, (3, 3), input_shape=input_shape))   
#Image should be reduces. Will have  alot of data which is not required
model.add(Activation('relu')) 
model.add(MaxPooling2D(pool_size = (2, 2)))

model.summary()

model.add(Conv2D(32, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
#Doing same thing but extracting 64 features from it
model.add(Conv2D(64, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
#Flattening image from 2D to 1D
model.add(Flatten())
#Need hidden layers that activate with given data and then gives output 64. It takes 64 features as input
model.add(Dense(64))  
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(1))
#Output layer function is sigmoid
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
#Saving model into .h5 file
model.save('model.h5')

img_pred = image.load_img('nikiopen.jpg', target_size = (150, 150))
img_pred = image.img_to_array(img_pred)
img_pred = np.expand_dims(img_pred, axis = 0)

rslt = model.predict(img_pred)
print (rslt)
if rslt[0][0] == 1 :
    prediction = "open"
else :
    prediction = "closed"

print (prediction)
