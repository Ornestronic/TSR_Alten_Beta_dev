from fastai import *
import cv2
import pyglet 
import numpy as np
import os, threading

class my_tsr :

  def _init_ (self):
    #Initialisation des variables de classe
    self.im = []
    self.tfm = get_transforms(do_flip=False, flip_vert=False, max_rotate=2.0, max_zoom=1.1, max_lighting=0.68, p_affine=0.55, p_lighting=0.75)
    self.learn = load_learner(path, 'sque1_2.pkl') #path to define
    self.speed = [0, 1, 2, 3, 4, 5, 6, 7, 8]
    self.stop = [14, 49]
    self.pedestrian = [27, 28, 48, 50]
    self.turn = [19, 20, 21, 33, 34]
    self.tabImg = []

  def detection(self, frame):
    frame = cv2.resize(frame,(640,480))
    self.im = frame[0:80, 0:640]
    gray = cv2.cvtColor(self.im, cv2.COLOR_BGR2GRAY)
    crop = Traffic.detectMultiScale(gray, scaleFactor=1.12, minNeighbors=4, minSize=(35,35), maxSize=(45,45))
    return crop

  def cropping(self, coords):
      X, Y, w, h = coords
      H, W, _ = self.im.shape
      X_1, X_2 = (max(0, X), min(X + int(w), W))
      Y_1, Y_2 = (max(0, Y), min(Y + int(h), H))
      image = self.im[Y_1:Y_2, X_1:X_2]
      return image

  def prediction(self, crop_image):
      img = Image(pil2tensor(crop_image, np.float32).div_(255))
      img = img.apply_tfms(self.tfm[1], size=64, resize_method=3)
      pred = learn.predict(img)
      prob = float(np.array(max(pred[2])))  
      if prob >= 0.92 :
        p = os.path.sep.join(["test","{}.png".format(i)])
        cv2.imwrite(p,crop_image)
        Id = int(str(pred[0]))
        print(prob,Id)        
      else :
        Id = 'none'
      return Id

  def displaySound(self, ID):

    if ID in self.speed:
      track ='speed'
    elif ID in self.stop:
      track = 'stop'
    elif ID in self.pedestrian:
      track = 'ped'
    elif ID in self.turn:
      track = 'turn'
    #if you want to play the sound just uncomment these two lines
    #music= pyglet.resource.media('{}.wav'.format(track), streaming = False)
    #music.play()
    return track

  def displayImg(self, ID):
    if ID in self.speed:
       speed ='Meta2/{}.png'.format(ID)
    elif ID in self.stop:
       stop ='Meta2/{}.png'.format(ID)
    elif ID in self.pedestrian:
       ped ='Meta2/{}.png'.format(ID)
    elif ID in self.turn:
       turn = stop ='Meta/{}.png'.format(ID)
    return [speed, stop, ped, turn]


  def display(self, ID):
    p1 = threading.Thread(target=displaySound, args=[ID])
    p2 = threading.Thread(target=displayImg, args=[ID])
    p1.start()
    p2.start()
    p1.join()
    p2.join()