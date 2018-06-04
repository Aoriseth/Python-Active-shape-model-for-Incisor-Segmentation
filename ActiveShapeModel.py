
# coding: utf-8

# In[174]:


import os
import glob
import cv2
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial import procrustes
from sklearn.decomposition import PCA
import math


def get_range_of(i, pca):
    eigenvalues = get_eigenvalues(pca)
    bound = 3*math.sqrt(eigenvalues[i])
    return bound

def generate_model_point(b, pca):
    eigenvectors = get_eigenvectors(pca)
    P = eigenvectors.transpose(1,0)
    #b = get_eigenvalues(pca)
    xm = get_mean(pca)
    
    x =  np.dot(P,b)
    return x + xm

def transform(p, pose_param):
    Txy, scale, rad = pose_param
    rotation = np.array([[math.cos(rad), -math.sin(rad)],[ math.sin(rad), math.cos(rad)]])
    return Txy + np.dot(s*rotation, p)

def inv_transform(p, pose_param):
    Txy, scale, rad = pose_param
    rotation = np.array([[math.cos(rad), math.sin(rad)],[ -math.sin(rad), math.cos(rad)]])
    return np.dot(rotation/s, p-Txy)

def project_to_tangent_plane(y, pca):
    xm = pca.mean_
    return y / np.dot(y,xm)

def update_model_param(y, pca):
    xm = pca.mean_
    PT = get_eigenvectors(pca)
    return np.dot(PT, y - xm)


#protocol 1
def match_model_points(pca):
    b = np.zeros(len(pca.components_)) 
    
    while !converged:
        
        x = generate_model_point(b, pca)

        pose_param = find_pose_param()

        y = inv_transform(Y,pose_param)

        y = project_to_tangent_plane(y, pca)

        b = update_model_param(y, pca)


        
        



# In[1]:


if __name__ == "__main__":
    #main

