# -*- coding: utf-8 -*-
"""
Created on Tue Feb 13 11:32:20 2018

@author: KARHAUSE
"""
from threading import Thread
import numpy as np
import cv2

class CFrame:
    def __init__(self, src=0):
        self.width = 0
        self.height = 0
        self.frame = None
        self.ret = False
        self.capture = cv2.VideoCapture(src)
        if self.capIsOpen():
            self.getDimension()
        self.stopped = False
        self.ret, self.frame = self.capture.read()
        pass
    
    def capIsOpen(self):
        self.srcIsOpen = self.capture.isOpened()
        return self.srcIsOpen
    
    def start(self):
        # start the thread to read frames from the video stream
        Thread(target=self.update, args=()).start()
        return self

    def update(self):
        # keep looping infinitely until the thread is stopped
        while True:
            # if the thread indicator variable is set, stop the thread
            if self.stopped:
                return
            # otherwise, read the next frame from the stream
            self.ret, self.frame = self.capture.read()
    
    def getDimension(self):
        self.width = self.capture.get(3)
        self.height = self.capture.get(4)
        
    def setDimension(self, width, height):
        self.capture.set(3, width)
        self.capture.set(4, height)
        
    def getWidth(self):
        return self.width
    
    def getHeight(self):
        return self.height
    
    def read(self, show=False):
        if self.ret:
            if show:
                self.showFrame()
            return self.frame
        else:
            return None
        
    def showFrame(self):
        if self.ret:
            cv2.imshow('frame', self.frame)
    
    def close(self):
        if self.stopped == False:
            self.stop()
        self.capture.release()
        
    def stop(self):
        # indicate that the thread should be stopped
        self.stopped = True