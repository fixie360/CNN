#python 3.6
#keras x.xx
#tensorflow x.x


#Import Keras libraries and packages
from keras.models import Sequential
from keras.models import model_from_json
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing import image
import numpy as np
#And if you want to check that the GPU is correctly detected, start your script with:
import tensorflow as tf
sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))


#Create object of the Sequential class
model = Sequential()

#Add layers to model object, building the CNN model
#Convolution
model.add(Conv2D(32, (3, 3), input_shape = (64, 64, 3), activation = 'relu'))
#Pooling
model.add(MaxPooling2D(pool_size = (2, 2)))
#Flatten
model.add(Flatten())
#Dense
model.add(Dense(units = 128, activation = 'relu'))
#Initialise output layer
model.add(Dense(units = 1, activation = 'sigmoid'))

#Compile CNN from model
model.compile(optimizer= 'adam', loss ='binary_crossentropy', metrics = ['accuracy'])

#Pre-process images
train_datagen = ImageDataGenerator(rescale = 1./255,
shear_range = 0.2,
zoom_range = 0.2,
horizontal_flip = True)
test_datagen = ImageDataGenerator(rescale = 1./255)
training_set = train_datagen.flow_from_directory('data/training_set',
target_size = (64, 64),
batch_size = 32,
class_mode = 'binary')
test_set = test_datagen.flow_from_directory('data/test_set',
target_size = (64, 64),
batch_size = 32,
class_mode = 'binary')

#Fit the data to the CNN
model.fit_generator(training_set,
#good figure is 8000
steps_per_epoch = 8000,
#good figure is 25
epochs = 25,
validation_data = test_set,
validation_steps = 2000)

#Make new predictions from the trained model
test_image = image.load_img('data/single_prediction/cat_or_dog_1.jpg',
target_size = (64, 64))
test_image = image.img_to_array(test_image)
test_image = np.expand_dims(test_image, axis = 0)
result = model.predict(test_image)
training_set.class_indices
#flow_from_directory method takes an argument 'classes' that can define a list of optional sub-directories you wish to be indexed in the order you define. However, if no value is provided, each sub directory will be inferred as a class, and the classes will be order alphanumerically
if result[0][0] == 1:
    prediction = 'dog'
else:
    prediction = 'cat'

print (prediction)

# serialize model to JSON
model_json = model.to_json()
with open("model.json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
model.save_weights("model.h5")
print("Saved model to disk")
