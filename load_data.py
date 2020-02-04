#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 30 14:54:12 2020

@author: max
"""

import glob
import numpy as np
import imageio
from sklearn.model_selection import train_test_split
from skimage.transform import resize

def main():
    
    
    categories = sorted(glob.glob('*'))
    images = []
    labels = []
    
    
    for category in categories:
        for test_filename in glob.glob(category+'/*'):
            image = np.array(imageio.imread(test_filename))
            # resize image
            image = resize(image, (224,224,3))
            images.append(image)
            labels.append(category)
            
    # one hot encode the labelse
    
    oneHotEncoder = dict()
    for idx,i in enumerate(categories):
        encodedVec = np.zeros(2)
        encodedVec[idx] = 1
        
        oneHotEncoder[i] = encodedVec
    
    
    
    encoded_labels=[]
    for i in labels:
        encoded_labels.append(oneHotEncoder[i])
    
    
    X_train, X_test, y_train, y_test = train_test_split(images,encoded_labels)
    
    X_train = np.array(X_train)
    y_train = np.array(y_train)
    X_test = np.array(X_test)
    y_test = np.array(y_test)

    return(X_train, X_test, y_train, y_test)