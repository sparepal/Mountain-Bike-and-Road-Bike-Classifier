# -*- coding: utf-8 -*-
"""
Created on Tue Mar 19 22:34:43 2019

@author: vikhy
"""

# Part 3 - Making new predictions
import numpy as np
from keras.preprocessing import image

#Appending the test set directory and file names to load later
from os import listdir
from os.path import isfile, join
onlyfiles = [r'C:\Files\Datasets\bikes\test_set\mountain_bikes'+'\\'+f for f in listdir(r'C:\Files\Datasets\bikes\test_set\mountain_bikes') if isfile(join(r'C:\Files\Datasets\bikes\test_set\mountain_bikes', f))]
onlyfiles += [r'C:\Files\Datasets\bikes\test_set\road_bikes'+'\\'+f for f in listdir(r'C:\Files\Datasets\bikes\test_set\road_bikes') if isfile(join(r'C:\Files\Datasets\bikes\test_set\road_bikes', f))]

#Shuffling the test set
from random import shuffle
shuffle(onlyfiles)

def predict_display(filename):
    test_image = image.load_img(filename, target_size = (64, 64))
    test_image = image.img_to_array(test_image)
    test_image = np.expand_dims(test_image, axis = 0)
    result = classifier.predict(test_image)
    if result[0][0] == 0:
        prediction = 'mountain_bike'
    else:
        prediction ='road_bike'
   
    return prediction 

#Getting the results
result=[]
for filename in onlyfiles:
    result.append(predict_display(filename))
    

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
plt.figure()
%matplotlib inline
for i in range(0, len(onlyfiles)):
    img=mpimg.imread(onlyfiles[i])
    plt.imshow(img)
    plt.title(result[i])
    #plt.waitforbuttonpress()
    plt.show()
    
    
