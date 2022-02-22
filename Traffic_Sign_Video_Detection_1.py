from fastai.vision import *
import numpy as np
import cv2
import os

learn = load_learner("dataset", 'sque1_2.pkl')
    
def detection():
        
        global im
        global frame
        global ret
        ret, frame = cap.read()
        frame = cv2.resize(frame,(640,480))
        im = frame[0:80, 0:640]
        gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
        crop = Traffic.detectMultiScale(gray, scaleFactor=1.12,minNeighbors=4,minSize=(35,35),maxSize=(45,45))


        return crop
            
def cropping(coords):
    
        X, Y, w, h = coords
        H, W, _ = im.shape
        X_1, X_2 = (max(0, X), min(X + int(w), W))
        Y_1, Y_2 = (max(0, Y), min(Y + int(h), H))
        image = im[Y_1:Y_2, X_1:X_2]
        
        return image
            
def prediction(crop_image):
        
        img = Image(pil2tensor(crop_image, np.float32).div_(255))
        pred = learn.predict(img)
        prob = float(np.array(max(pred[2])))
        
        if prob >= 0.92 :
            Id = int(str(pred[0]))
            print(prob,Id)        
        else :
            Id = 'none'
        
        return Id
    
def display(ID, frame):
        if ID is not 'none' :
            img = cv2.imread("Meta/{}.png".format(ID))
            img = cv2.resize(img,(30,30))
            frame[0:30,0:30] = img
        return frame

def arg_parse():
        
        parser = argparse.ArgumentParser(description='First approach')

        parser.add_argument("--video", dest = 'video', help = "Video to run detection upon",default = "video.avi", type = str)

        return parser.parse_args()

if __name__ == '__main__':

        args = arg_parse()        
    
        Traffic = cv2.CascadeClassifier("cascade_4.xml")

        cap = cv2.VideoCapture(args.video)

        
        frame_width = int(cap.get(3))
        frame_height = int(cap.get(4))
    
        out = cv2.VideoWriter('test_1.avi',cv2.VideoWriter_fourcc('M','J','P','G'), 30, (frame_width,frame_height))

        while(cap.isOpened()):

                if ret :

                        crop = detection()
    
                        for coords in crop:
        
                                image = cropping(coords)
    
                                ID = prediction(image)
    
                                frame = display(ID, frame)

                        out.write(frame)

                else :
                        break

                        
    

