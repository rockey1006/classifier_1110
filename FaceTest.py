import os
import cv2
import sys
import dlib
import tensorflow.keras.backend as K
from tensorflow.keras.preprocessing.image import load_img,img_to_array
from tensorflow.keras.applications.imagenet_utils import preprocess_input
import numpy as np
import matplotlib.pyplot as plt

class FaceTest(object):
    #def __init__(self, *args, **kwargs):
        #self.dnnFaceDetector=dlib.cnn_face_detection_model_("mmod_human_face_detector.dat")

    def test_facerecog_in_dir(self,dnnFaceDetector, vgg_face, classifier_model, person_rep, dirname):
       for img_name in os.listdir(dirname):
          if img_name=='crop_img.jpg':
            continue
          # Load Image
          img=cv2.imread(dirname + '/'+img_name)
          gray=cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
          # Detect Faces
          rects= dnnFaceDetector(gray,1)
          left,top,right,bottom=0,0,0,0
          for (i,rect) in enumerate(rects):
            # Extract Each Face
            left=rect.rect.left() #x1
            top=rect.rect.top() #y1
            right=rect.rect.right() #x2
            bottom=rect.rect.bottom() #y2
            width=right-left
            height=bottom-top
            img_crop=img[top:top+height,left:left+width]
            cv2.imwrite('Test_Images/crop_img.jpg',img_crop)
    
            # Get Embeddings
            crop_img=load_img('Test_Images/crop_img.jpg',target_size=(224,224))
            #crop_img = img_crop
            crop_img=img_to_array(crop_img)
            crop_img=np.expand_dims(crop_img,axis=0)
            crop_img=preprocess_input(crop_img)
            img_encode=vgg_face(crop_img)

            # Make Predictions
            embed=K.eval(img_encode)
            person=classifier_model.predict(embed)
            name=person_rep[np.argmax(person)]
            os.remove('Test_Images/crop_img.jpg')
            cv2.rectangle(img,(left,top),(right,bottom),(0,255,0), 2)
            img=cv2.putText(img,name,(left,top-10),cv2.FONT_HERSHEY_SIMPLEX,1,(255,0,255),2,cv2.LINE_AA)
            img=cv2.putText(img,str(np.max(person)),(right,bottom+10),cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,0,255),1,cv2.LINE_AA)
          # Save images with bounding box,name and accuracy 
          cv2.imwrite('Predictions/'+img_name,img)
          self.plot(img)

    def plot(self,img):
      plt.figure(figsize=(8,4))
      plt.imshow(img[:,:,::-1])
      plt.show()

