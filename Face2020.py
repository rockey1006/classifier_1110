from FaceDetect import FaceDetect
from FaceModel import FaceModel
from PrepareDataset import PrepareDataset
from tensorflow.keras.models import Sequential,Model
from FaceTest import FaceTest
import tensorflow as tf
import subprocess as sp
import os
import sys

def main():
    # to install dlib, first instal cmake in visual studio (under c++, install options)
    # set path to cmake
    # pip install cmake
    # pip install dlib

    facemodel = FaceModel()
    faceTest = FaceTest()   # for testing a folder of face pictures
    faceDetect = FaceDetect()
    dnnfacedetector = faceDetect.detect_face()  # initialize dnnface detector first
                                                # otherwise it generates memory error in gpu
    # Label names for class numbers
    model_file_name = '/content/vgg_face_weights/vgg_face_weights.h5'
    model = facemodel.create_and_load_face_model(model_file_name)
    # Remove last Softmax layer and get model upto last flatten layer 
    # with outputs 2622 units 
    vgg_face = Model(inputs=model.layers[0].input,outputs=model.layers[-2].output)
    print('model loaded-------------')
    
    # -------prepare dataset-----------
    pd = PrepareDataset()
    x_train, y_train, x_test, y_test, person_rep = pd.prepare_data('',vgg_face)
    #print(model.summary())

    #train the classifier
    classifier_model = facemodel.train_model(x_train,y_train, x_test,y_test)
    #tf.keras.models.save_model(classifier_model,'classifier_model.h5')
    
    #classifier_model=tf.keras.models.load_model('classifier_model.h5')
    
    faceTest.test_facerecog_in_dir(dnnfacedetector,vgg_face,classifier_model,person_rep,'/content/images')
    print('done..')


if __name__ == "__main__":
    sys.exit(int(main() or 0))
