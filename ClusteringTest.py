# -*- coding: utf-8 -*-
"""
Created on Mon Apr 30 08:31:50 2018

@author: KARHAUSE
"""

import neural.clustering.KMeans as km
import neural.testdata.testdata as td

clusterer = km.CKMeans(3)
clusterer.train(td.data)