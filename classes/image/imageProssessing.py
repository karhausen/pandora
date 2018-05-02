# -*- coding: utf-8 -*-
"""
Created on Tue Feb 13 11:33:07 2018

@author: KARHAUSE
"""

import numpy as np
import cv2

class CImageProcessing:
    def __init__(self, img=None):
        self.image = None
        self.edges = None
        self.gray = None
        self.haarcascadePath = 'C:/Anaconda3/Library/etc/haarcascades/'
        self.face_cascade = cv2.CascadeClassifier(self.haarcascadePath + 'haarcascade_frontalface_default.xml')
        self.eye_cascade = cv2.CascadeClassifier(self.haarcascadePath + 'haarcascade_eye.xml')
        pass
    
    def getEdges(self, img):
        self.image = img
        self.edges = cv2.Canny(self.image, 100, 200)
        return self.edges
    
    def getGray(self, img):
        self.image = img
        self.gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        return self.gray
        pass
    
    def getFaces(self, img):
        self.roi_gray = []
        self.roi_color = []
        self.image = img
        self.getGray(img)
        self.faces = self.face_cascade.detectMultiScale(self.gray, 1.3, 5)
        for (x,y,w,h) in self.faces:
            #img = cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
            self.roi_gray.append( self.gray[y:y+h, x:x+w] )
            self.roi_color.append( self.image[y:y+h, x:x+w] )
        return self.faces
        pass
    
    def boxes(self, img, face, color=(255,0,0)):
        x = face[0]
        y = face[1]
        w = face[2]
        h = face[3]
        img = cv2.rectangle(img, (x,y), (x+w,y+h), color, 2)
        return img
    
    def getFacesAsImg(self, img):
        self.getFaces(img)
        return self.roi_color
    
    def getEyes(self, img):
        #self.image = img
        self.eyes = self.eye_cascade.detectMultiScale( img )
        return self.eyes
