import os
from tensorflow.keras.preprocessing.image import load_img,img_to_array
import numpy as np
from tensorflow.keras.applications.imagenet_utils import preprocess_input
import tensorflow.keras.backend as K

class PrepareDataset(object):
    def prepare_data(self, path, vgg_face):
        # Prepare Train Data
        x_train=[]
        y_train=[]
        person_rep=dict()
        traindir = path+'/content/new_A1/'
        
        person_folders=sorted(os.listdir(traindir))
       
        for i,person in enumerate(person_folders):
          person_rep[i]=person
          image_names=sorted(os.listdir(traindir+ person+'/'))
          for image_name in image_names:
            img=load_img(traindir + person+'/'+image_name,target_size=(224,224))
            img=img_to_array(img)
            img=np.expand_dims(img,axis=0)
            img=preprocess_input(img)
            img_encode=vgg_face(img)
            x_train.append(np.squeeze(K.eval(img_encode)).tolist())
            y_train.append(i)
    
        # Prepare Test Data
        x_test=[]
        y_test=[]
        person_folders=os.listdir(path+'/content/new_A2/')
        testdir = path+'/content/new_A2/'
        person_folders=sorted(os.listdir(traindir))
      
       
        test_image_names=sorted(os.listdir(testdir))
        for i,person in enumerate(person_folders):
          person_rep[i]=person
          image_names=os.listdir(testdir+ person+'/')
          for image_name in image_names:
            img=load_img(testdir+person+'/'+image_name,target_size=(224,224))
            img=img_to_array(img)
            img=np.expand_dims(img,axis=0)
            img=preprocess_input(img)
            img_encode=vgg_face(img)
            x_test.append(np.squeeze(K.eval(img_encode)).tolist())
            y_test.append(i)
        
        x_train=np.array(x_train) 
        y_train=np.array(y_train)
        x_test=np.array(x_test) 
        y_test=np.array(y_test)
        print(x_train.shape)
        print(y_train.shape)
        return x_train, y_train, x_test, y_test, person_rep


