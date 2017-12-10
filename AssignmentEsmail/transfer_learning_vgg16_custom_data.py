

import numpy as np
import os
os.chdir("/home/spider/Documents/machineLearningAssignment/AssignmentEsmail")
import time
from vgg16 import VGG16
from keras.preprocessing import image
from imagenet_utils import preprocess_input, decode_predictions
from keras.layers import Dense, Activation, Flatten
from keras.layers import merge, Input
from keras.models import Model
from keras.utils import np_utils

from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras import backend as K

from sklearn.utils import shuffle
from sklearn.cross_validation import train_test_split

#img_path = 'elephant.jpg'
#img = image.load_img(img_path, target_size=(224, 224))
#x = image.img_to_array(img)
#print (x.shape)
#x = np.expand_dims(x, axis=0)
#print (x.shape)
#x = preprocess_input(x)
#print('Input image shape:', x.shape)

#img_path = 'cat.jpg'
#img = image.load_img(img_path, target_size=(224, 224))
#x = image.img_to_array(img)
#print (x.shape)
#x = np.expand_dims(x, axis=0)
#print (x.shape)
#x = preprocess_input(x)
#print('Input image shape:', x.shape)

# Loading the training data
PATH = os.getcwd()
# Define data path
data_path = PATH + '/data'
data_dir_list = os.listdir(data_path)

img_data_list=[]

for dataset in data_dir_list:
	img_list=os.listdir(data_path+'/'+ dataset)
	print ('Loaded the images of dataset-'+'{}\n'.format(dataset))
	for img in img_list:
		img_path = data_path + '/'+ dataset + '/'+ img 
		img = image.load_img(img_path, target_size=(224, 224))
		x = image.img_to_array(img)
		x = np.expand_dims(x, axis=0)
		x = preprocess_input(x)
#		x = x/255
		print('Input image shape:', x.shape)
		img_data_list.append(x)

img_data = np.array(img_data_list)
#img_data = img_data.astype('float32')
print (img_data.shape)
img_data=np.rollaxis(img_data,1,0)
print (img_data.shape)
img_data=img_data[0]
print (img_data.shape)

imlist = os.listdir(data_path + "/monet")


# Define the number of classes
num_classes = 4
num_of_samples = img_data.shape[0]
labels = np.ones((num_of_samples,),dtype='int64')

labels[0:197]=0
labels[197:393]=1
labels[393:589]=2
labels[589:785]=3

#names = ['cats','dogs','horses','humans']
names = ['renoir', 'manet', 'degas', 'monet']
	  
# convert class labels to on-hot encoding
Y = np_utils.to_categorical(labels, num_classes)

#Shuffle the dataset
x,y = shuffle(img_data,Y, random_state=2)
# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=2)


#########################################################################################
# Custom_vgg_model_1
#Training the classifier alone
#image_input = Input(shape=(224, 224, 3))
#
#model = VGG16(input_tensor=image_input, include_top=True, weights='imagenet')
#model.summary()
#last_layer = model.get_layer('fc2').output
##x= Flatten(name='flatten')(last_layer)
#out = Dense(num_classes, activation='softmax', name='output')(last_layer)
#custom_vgg_model = Model(image_input, out)
#custom_vgg_model.summary()
#
#for layer in custom_vgg_model.layers[:-1]:
#	layer.trainable = False
#
#custom_vgg_model.layers[3].trainable
#
#custom_vgg_model.compile(loss='categorical_crossentropy',optimizer='rmsprop',metrics=['accuracy'])
#
#
#t=time.time()
##	t = now()
#hist = custom_vgg_model.fit(X_train, y_train, batch_size=4, epochs=1, verbose=1, validation_data=(X_test, y_test))
#print('Training time: %s' % (t - time.time()))
#(loss, accuracy) = custom_vgg_model.evaluate(X_test, y_test, batch_size=10, verbose=1)
#
#print("[INFO] loss={:.4f}, accuracy: {:.4f}%".format(loss,accuracy * 100))


####################################################################################################################

#Training the feature extraction also

image_input = Input(shape=(224, 224, 3))

model = VGG16(input_tensor=image_input, include_top=True,weights='imagenet')

model.summary()

last_layer = model.get_layer('block5_pool').output
x= Flatten(name='flatten')(last_layer)
x = Dense(128, activation='relu', name='fc1')(x)
x = Dense(128, activation='relu', name='fc2')(x)
out = Dense(num_classes, activation='softmax', name='output')(x)
custom_vgg_model2 = Model(image_input, out)
custom_vgg_model2.summary()

# freeze all the layers except the dense layers
for layer in custom_vgg_model2.layers[:-3]:
	layer.trainable = False

custom_vgg_model2.summary()

custom_vgg_model2.compile(loss='categorical_crossentropy',optimizer='adadelta',metrics=['accuracy'])

t=time.time()
#	t = now()
hist = custom_vgg_model2.fit(X_train, y_train, batch_size=4, epochs=20, verbose=1, validation_data=(X_test, y_test))
print('Training time: %s' % (t - time.time()))
(loss, accuracy) = custom_vgg_model2.evaluate(X_test, y_test, batch_size=10, verbose=1)

print("[INFO] loss={:.4f}, accuracy: {:.4f}%".format(loss,accuracy * 100))

#custom_vgg_model2.save_weights("model12epoch.h5")

#Predicting on w.e image
#img_path2 = 'outlier.png'
#img_path2 = 'degas2.jpg'
#img_path2 = 'manet.png'

#img_path2 = 'degas.png'
#
#img_path2 = 'renoir.png'

img2 = image.load_img(img_path2, target_size=(224, 224))
x1 = image.img_to_array(img2)
x1 = np.expand_dims(x1, axis=0)
x1 = preprocess_input(x1)
print('Input image shape:', x1.shape)
preds1 = custom_vgg_model2.predict(x1)
print('Predicted:', preds1)

index = np.argmax(preds1)
label = names[index]
print("Predicted:", label)


custom_vgg_model2.summary()
custom_vgg_model2.layers[-1].get_config()
#%%
import matplotlib.pyplot as plt
# visualizing losses and accuracy
train_loss=hist.history['loss']
val_loss=hist.history['val_loss']
train_acc=hist.history['acc']
val_acc=hist.history['val_acc']
xc=range(12)

plt.figure(1,figsize=(7,5))
plt.plot(xc,train_loss)
plt.plot(xc,val_loss)
plt.xlabel('num of Epochs')
plt.ylabel('loss')
plt.title('train_loss vs val_loss')
plt.grid(True)
plt.legend(['train','val'])
#print plt.style.available # use bmh, classic,ggplot for big pictures
plt.style.use(['classic'])

plt.figure(2,figsize=(7,5))
plt.plot(xc,train_acc)
plt.plot(xc,val_acc)
plt.xlabel('num of Epochs')
plt.ylabel('accuracy')
plt.title('train_acc vs val_acc')
plt.grid(True)
plt.legend(['train','val'],loc=4)
#print plt.style.available # use bmh, classic,ggplot for big pictures
plt.style.use(['classic'])
