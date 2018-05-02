# -*- coding: utf-8 -*-
"""
Created on Tue Feb 13 11:31:23 2018

@author: KARHAUSE
"""

import numpy as np
import cv2

class CImage:
    def __init__(self):
        self.img = None
        pass
    
    def getImage(self, src):
        self.img = cv2.imread(src,1)
        return self.img
    
    def showImage(self):
        cv2.imshow('image', self.img)