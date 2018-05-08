
# coding: utf-8

# In[141]:


import os
import glob
import cv2
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial import procrustes
from sklearn.decomposition import PCA

def load_files(dir_images):
    
    inputList = []
    
    inputNames = glob.glob(dir_images)
    
    for inputName in inputNames:
    
        inputFile = cv2.imread(inputName, 0)
        inputList.append(inputFile)

    return inputList


def load_landmarks():
    
    dir_landmarks = "_Data\Landmarks\original\*.txt"
    inputNames = glob.glob(dir_landmarks)
    
    inputList = np.empty([len(inputNames), 40, 2])
    
    for i, inputName in enumerate(inputNames):
    
        with open(inputName) as f:
            content = f.readlines()
            
        content = [float(x.strip()) for x in content] 
        
        inputList[i] = np.asarray(content).reshape(40,2)
        
    inputList = inputList.reshape(14,8, 40, 2)
        
    return inputList
    
    
def show(img):

    plt.imshow(img)
    plt.show()
    
    
def procrustes_analysis(landmarks):
    
    mean = np.mean(landmarks,0)
    landmarks_std = np.empty_like(landmarks)
    
    for i, landmark in enumerate(landmarks):
        
        mean_std, landmark_std, disp = procrustes(mean, landmark)
        landmarks_std[i] = landmark_std
    
    return landmarks_std

def total_procrustes_analysis(all_landmarks):
    #allign shapes in their set
    
    all_landmarks_std = np.empty_like(all_landmarks)
    
    for i, landmarks in enumerate(all_landmarks):
        
        landmarks_std = procrustes_analysis(landmarks)
        all_landmarks_std[i] = landmarks_std
        
    return all_landmarks_std


def PCA_analysis(data, number_of_components):
    
    data = data.reshape(-1,80)
    pca = PCA(n_components= number_of_components)
    pca.fit(data)
    
    return pca

def PCA_reconstruction(pca, data):
    
    projections = pca.transform(data)
    reconstructions = pca.inverse_transform(projections)
    
    return reconstructions


######### Visualization #########
    
def show_tooth_points(landmark, show):
    
    plt.plot(landmark[:,0], landmark[:,1], 'ro')
    
    if show:
        plt.show()
        
def show_teeth_points(landmarks):
    
    plt.figure()
    n = len(landmarks)
    hn = int(n/2)
   
    print('Showing Teeth Landmarks')

    for i, landmark in enumerate(landmarks):
        plt.subplot(2, hn, i+1)
        plt.xticks(())
        plt.yticks(())   
        plt.plot(landmark[:,0], landmark[:,1], 'ro')
     
    plt.show(block=False)
    
def show_PCA(pca):
    
    for i in range(len(pca.components_)):
        eig = pca.components_[i].reshape(40,2)
        plt.imshow(eig)
        plt.show()
        
        
def show_PCAs(pca):
    
    plt.figure()
    n = len(pca.components_)
    hn = int(n/2)
   
    print('Showing PCA\'s')

    for i, vector in enumerate(pca.components_):
        vector= vector.reshape(40,2)
        plt.subplot(2, hn, i+1)
        plt.xticks(())
        plt.yticks(())   
        show_tooth_points(vector, False)
     
    plt.show()


# In[142]:


#main

dir_radiographs = "_Data\Radiographs\*.tif"
radiographs = load_files(dir_radiographs)

dir_segmentations = "_Data\Segmentations\*.png"
segmentations = load_files(dir_segmentations)

all_landmarks = load_landmarks()
show_teeth_points(all_landmarks[0])

all_landmarks_std = total_procrustes_analysis(all_landmarks)
show_teeth_points(all_landmarks_std[0])

pca = PCA_analysis(all_landmarks_std[0], 8)
show_PCAs(pca)

