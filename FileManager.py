
# coding: utf-8

# In[7]:


import os
import glob
import cv2
import matplotlib.pyplot as plt
from scipy.spatial import procrustes
import numpy as np
import math


def load_radiographs():
    dir_radiographs = "_Data\Radiographs\*.tif"
    return load_files(dir_radiographs)


def load_segmentations():
    dir_segmentations = "_Data\Segmentations\*.png"
    return load_files(dir_segmentations)


def load_files(dir_images):
    
    inputList = []
    
    inputNames = glob.glob(dir_images)
    
    for inputName in inputNames:
    
        inputFile = cv2.imread(inputName, 0)
        inputList.append(inputFile)

    return inputList


def load_landmarks():
    
    dir_landmarks = "_Data/Landmarks/original/*.txt"
    inputNames = glob.glob(dir_landmarks)
    
    inputList = np.empty([len(inputNames), 40, 2])
    
    for i, inputName in enumerate(inputNames):
    
        with open(inputName) as f:
            content = f.readlines()
            
        content = [float(x.strip()) for x in content] 
        
        inputList[i] = np.asarray(content).reshape(40,2)
        
    inputList = inputList.reshape(14,8, 40, 2)
        
    return inputList


def load_landmarks_std():
    
    all_landmarks = load_landmarks()
    return total_procrustes_analysis(all_landmarks)

def mean_landmarks(landmarks):
    return np.mean(landmarks,0)
    
def procrustes_analysis(landmarks):
    
    mean = np.mean(landmarks,0)
    landmarks_std = np.empty_like(landmarks)
    
    for i, landmark in enumerate(landmarks):
        
        mean_std, landmark_std, disp = procrustes(mean, landmark)
        landmarks_std[i] = landmark_std
    
    return landmarks_std

def total_procrustes_analysis(all_landmarks):
    #allign shapes in their set
    
    all_landmarks = np.transpose(all_landmarks, (1,0,2,3))
    all_landmarks_std = np.empty_like(all_landmarks)
    
    for i, landmarks in enumerate(all_landmarks):
        
        landmarks_std = procrustes_analysis(landmarks)
        all_landmarks_std[i] = landmarks_std
        
    all_landmarks_std = np.transpose(all_landmarks_std, (1,0,2,3))
    return all_landmarks_std

    
######### Visualization #########

def show(img):

    plt.imshow(img)
    plt.show()
    
def show_tooth_points(landmark, show=True):
    
    plt.plot(landmark[:,0], landmark[:,1], 'ro')
    
    if show:
        plt.show()
        
def show_teeth_points(landmarks):
    
    # plt.figure()
    n = len(landmarks)
    hn = int(n/2)
   
    print('Showing Teeth Landmarks')
    f, xplot = plt.subplots(2,hn,figsize=(5, 5))

    for i, landmark in enumerate(landmarks):
        cursubplot = xplot[math.floor(i/hn),i-hn*(math.floor(i/hn))]
        cursubplot.plot(landmark[:,0], landmark[:,1], 'ro')
        # plt.subplot(2, hn, i+1)
        cursubplot.set_xticks(())
        cursubplot.set_yticks(())
        # cursubplot.xticks(())
        # cursubplot.yticks(())   
        # plt.plot(landmark[:,0], landmark[:,1], 'ro')
    plt.show()


# In[9]:


if __name__ == "__main__":
    #main

    dir_radiographs = "_Data/Radiographs/*.tif"
    radiographs = load_files(dir_radiographs)

    dir_segmentations = "_Data/Segmentations/*.png"
    segmentations = load_files(dir_segmentations)

    all_landmarks = load_landmarks()
    show_teeth_points(all_landmarks[0])
    
    all_landmarks_std = total_procrustes_analysis(all_landmarks)
    show_teeth_points(all_landmarks_std[:,0])
    
    all_landmarks_std = load_landmarks_std()
    show_teeth_points(all_landmarks_std[:,0])

