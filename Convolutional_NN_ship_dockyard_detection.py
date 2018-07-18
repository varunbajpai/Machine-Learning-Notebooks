
# coding: utf-8

# Importing Libraries in Keras, for setting up CNN for Ship Image Detection, We want to create a Sequential NN, Hence we import a Sequential model of the NN, Dense for setting up layers in the NN, 

# In[1]:


import json, sys, random
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Flatten, Activation
from keras.layers import Dropout
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.utils import np_utils
from keras.optimizers import SGD
import keras.callbacks
from matplotlib import pyplot as plt
from PIL import Image, ImageDraw 
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import plot_model
from keras.models import load_model


# In the function Below, This is used to get the image in the shape 80*80 basically cutting sections out of the image

# In[2]:


def cutting(x, y):
    area_study = np.arange(3*80*80).reshape(3, 80, 80)
    for i in range(80):
        for j in range(80):
            area_study[0][i][j] = picture_tensor[0][y+i][x+j]
            area_study[1][i][j] = picture_tensor[1][y+i][x+j]
            area_study[2][i][j] = picture_tensor[2][y+i][x+j]
    area_study = area_study.reshape([-1, 3, 80, 80])
    area_study = area_study.transpose([0,2,3,1])
    area_study = area_study / 255
    return area_study


# The function below will be used to dray boxes and detect objects in images, Ships in this case

# In[3]:


def show_object(x, y, acc, thickness=5):   
    for i in range(80):
        for ch in range(3):
            for th in range(thickness):
                picture_tensor[ch][y+i][x-th] = -1

    for i in range(80):
        for ch in range(3):
            for th in range(thickness):
                picture_tensor[ch][y+i][x+th+80] = -1
        
    for i in range(80):
        for ch in range(3):
            for th in range(thickness):
                picture_tensor[ch][y-th][x+i] = -1
        
    for i in range(80):
        for ch in range(3):
            for th in range(thickness):
                picture_tensor[ch][y+th+80][x+i] = -1


# In[4]:


#Downloading Dataset from Json File Available in the Download Section
f = open(r'shipsnet.json')
dataset = json.load(f)
f.close()
input_data = np.array(dataset['data']).astype('uint8')
output_data = np.array(dataset['labels']).astype('uint8')
print(input_data.shape)


# The DataSet Contains 3600 images and one image is represented by an array of 19200 elements
# [[],[],[],[]....[]] A 2-D list with inside lists = 3600 and each list of len 19200 (80x80x3(RGB))

# In[5]:


n_spectrum = 3 # color chanel (RGB)
weight = 80
height = 80
X = input_data.reshape([-1, n_spectrum, weight, height])
print(X[0])
print(X[0].shape)


# In[6]:


# This will create a vector for each image representing if there is an image containing ship or not wrt the
# corresponding Index
y = np_utils.to_categorical(output_data, 2)


# Randomly Shuffeling the Data to create Randomness so as to train the model better, then normalizing by dividing by 255 as each pixel is represented from a number between 0-255

# In[7]:


# shuffle all indexes
indexes = np.arange(2800)
np.random.shuffle(indexes)

X_train = X[indexes].transpose([0,2,3,1])
y_train = y[indexes]

# normalization
X_train = X_train / 255


# Creating the Network and training it using the data provided

# In[8]:


np.random.seed(42)

# network design
model = Sequential()
model.add(Conv2D(32, (3, 3), padding='same', input_shape=(80, 80, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2))) #40x40x32 is the total volume in this layer of convolution
model.add(Dropout(0.25))

model.add(Conv2D(32, (3, 3), padding='same', activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2))) #20x20x32 is the total volume in this layer of convolution
model.add(Dropout(0.25))

model.add(Conv2D(32, (3, 3), padding='same', activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2))) #10x10x32 is the total volume in this layer of convolution
model.add(Dropout(0.25))

model.add(Conv2D(32, (10, 10), padding='same', activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2))) #5x5x32 is the total volume in this layer of convolution
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(512, activation='relu'))
model.add(Dropout(0.5))

model.add(Dense(2, activation='softmax'))


# In[9]:


sgd = SGD(lr=0.01, momentum=0.9, nesterov=True)
model.compile(
    loss='categorical_crossentropy',
    optimizer=sgd,
    metrics=['accuracy'])

# training, Uncomment this line in case of training, since we already have a trained model we use that to 
#load
#model.fit(X_train,y_train,batch_size=32,epochs=18,validation_split=0.2,shuffle=True,verbose=2)


# Saving the Model for future Use As a Ship Detection Problem

# In[10]:


# from keras.models import load_model
# model.save('ship_dockyard_model.h5')


# In[11]:


model = load_model('ship_dockyard_model.h5')


# In[12]:


image = Image.open(r'test/sfbay_1.png')
test_image = image.load()
n_spectrum = 3              # Three layers of RGB
height = image.size[0]      # In order to iterate over the test_image to pick pixel by pixel along the height
width = image.size[1]       # Along the width with a stride of 1


#This will give us image object which can be used to create vector which will be fed into the NN
picture_vector = []
for chanel in range(n_spectrum):
    for y in range(height):
        for x in range(width):
            picture_vector.append(test_image[y, x][chanel])
picture_vector = np.array(picture_vector).astype('uint8')
picture_tensor = picture_vector.reshape([n_spectrum, height, width])


# In[ ]:


step = 10; coordinates = []
for y in range(int((height-(80-step))/step)):
    for x in range(int((width-(80-step))/step) ):
        area = cutting(x*step, y*step)
        print(area)
        result = model.predict(area)
        if result[0][1] > 0.90 and not_near(x*step,y*step, 88, coordinates):
            coordinates.append([[x*step, y*step], result])
            print(result)
            plt.imshow(area[0])
            plt.show()

