import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential,Model
from tensorflow.keras.layers import ZeroPadding2D,Convolution2D,MaxPooling2D
from tensorflow.keras.layers import Dense,Dropout,Softmax,Flatten,Activation,BatchNormalization

from tensorflow.keras.applications.imagenet_utils import preprocess_input
#import tensorflow.keras.backend as K

class FaceModel():
    def create_and_load_face_model(self, fname):
        # Define VGG_FACE_MODEL architecture
        model = Sequential()
        model.add(ZeroPadding2D((1,1),input_shape=(224,224, 3)))
        model.add(Convolution2D(64, (3, 3), activation='relu'))
        model.add(ZeroPadding2D((1,1)))
        model.add(Convolution2D(64, (3, 3), activation='relu'))
        model.add(MaxPooling2D((2,2), strides=(2,2)))
        model.add(ZeroPadding2D((1,1)))	
        model.add(Convolution2D(128, (3, 3), activation='relu'))
        model.add(ZeroPadding2D((1,1)))
        model.add(Convolution2D(128, (3, 3), activation='relu'))
        model.add(MaxPooling2D((2,2), strides=(2,2)))
        model.add(ZeroPadding2D((1,1)))
        model.add(Convolution2D(256, (3, 3), activation='relu'))
        model.add(ZeroPadding2D((1,1)))
        model.add(Convolution2D(256, (3, 3), activation='relu'))
        model.add(ZeroPadding2D((1,1)))
        model.add(Convolution2D(256, (3, 3), activation='relu'))
        model.add(MaxPooling2D((2,2), strides=(2,2)))
        model.add(ZeroPadding2D((1,1)))
        model.add(Convolution2D(512, (3, 3), activation='relu'))
        model.add(ZeroPadding2D((1,1)))
        model.add(Convolution2D(512, (3, 3), activation='relu'))
        model.add(ZeroPadding2D((1,1)))
        model.add(Convolution2D(512, (3, 3), activation='relu'))
        model.add(MaxPooling2D((2,2), strides=(2,2)))
        model.add(ZeroPadding2D((1,1)))
        model.add(Convolution2D(512, (3, 3), activation='relu'))
        model.add(ZeroPadding2D((1,1)))
        model.add(Convolution2D(512, (3, 3), activation='relu'))
        model.add(ZeroPadding2D((1,1)))
        model.add(Convolution2D(512, (3, 3), activation='relu'))
        model.add(MaxPooling2D((2,2), strides=(2,2)))
        model.add(Convolution2D(4096, (7, 7), activation='relu'))
        model.add(Dropout(0.5))
        model.add(Convolution2D(4096, (1, 1), activation='relu'))
        model.add(Dropout(0.5))
        model.add(Convolution2D(2622, (1, 1)))
        model.add(Flatten())
        model.add(Activation('softmax'))

        # Load VGG Face model weights
        #model.load_weights('vgg_face_weights.h5')
        model.load_weights(fname)
        return model

    def train_model(self, x_train, y_train, x_test, y_test):
        # Softmax regressor to classify images based on encoding 
        classifier_model=Sequential()	
        classifier_model.add(Dense(units=100,input_dim=x_train.shape[1],kernel_initializer='glorot_uniform'))		
        classifier_model.add(BatchNormalization())		
        classifier_model.add(Activation('sigmoid'))
        classifier_model.add(Dropout(0.2))
        #classifier_model.add(Dense(units=10,kernel_initializer='glorot_uniform'))
        #classifier_model.add(BatchNormalization())
        #classifier_model.add(Activation('tanh'))
        #classifier_model.add(Dropout(0.2))
        classifier_model.add(Dense(units=6000,kernel_initializer='he_uniform'))
        classifier_model.add(Activation('softmax'))
        classifier_model.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(),optimizer='nadam',metrics=['accuracy'])
        classifier_model.fit(x_train,y_train,epochs=1000)
        
        print("Evaluate on test data")
        results = classifier_model.evaluate(x_test, y_test, batch_size=128)
        print("test loss, test acc:", results)
        print("wrong data")
        predictions = classifier_model.predict_classes(x_test)
        wrong = np.where(predictions != y_test)
        print("positions:", wrong)
        





        return classifier_model
