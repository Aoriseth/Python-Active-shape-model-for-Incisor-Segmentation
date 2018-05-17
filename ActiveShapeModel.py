
# coding: utf-8

# In[8]:


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
    
    all_landmarks = np.transpose(all_landmarks, (1,0,2,3))
    all_landmarks_std = np.empty_like(all_landmarks)
    
    for i, landmarks in enumerate(all_landmarks):
        
        landmarks_std = procrustes_analysis(landmarks)
        all_landmarks_std[i] = landmarks_std
        
    all_landmarks_std = np.transpose(all_landmarks_std, (1,0,2,3))
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

def radiograph_preprocess(img):
    
    equ = cv2.equalizeHist(img)
    return equ

def radiograph_preprocess2(img):
    
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    cl1 = clahe.apply(img)
    return cl1


def gap_detection(img):
    
    h_proj = np.sum(img, axis=1)
    y = np.arange(img.shape[0])
    plt.plot(h_proj, y)
    plt.show()
    

def split(img, times):
    
    size, rem = np.divmod(img.shape[1] , times)
    splits = np.arange(0,img.shape[1], size)
    if rem > 0 :
        times += 1
    img_splitted = np.array((times, img.shape[0], size))
    length = len(splits)
    for i, split in enumerate(splits):
        if i == length - 1:
            img_splitted[i] = img[:,split:img.shape[1]-1
                                 ]
        img_splitted[i] = img[:,split:splits[i+1]]
        
    return img_splitted


def active_contour_match(img, init):
    snake = active_contour(gaussian(img, 3), init, alpha=0.015, beta=10, gamma=0.001)
    return snake
    
    
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
     
    plt.show()
    
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


# In[ ]:



# In[3]:



# In[11]:





# In[12]:





# In[13]:


import cv2
from scipy import interpolate

def intensity_prob(I, max_I, c=1):
    return c*(1-I/max_I)

def position_prob(Y, Yest, Sigma):
    t = (Y - Yest)**2 / (Sigma**2)
    return ( 1 / (np.sqrt(2*np.pi)*Sigma) ) * np.exp(-t)
    
    
def gap_valley_img(img, Yest, Sigma):
    
    img_copy = np.copy(img)
    h_proj = h_project(img)
    maxI = max(h_proj)
    pIY = np.empty_like(h_proj, dtype= np.float32)
    
    for Y, I in enumerate(h_proj):
        pI = intensity_prob(I,maxI)
        pY = position_prob(Y, Yest, Sigma)
        pIY[Y] = pI * pY
        
    gap = np.argmax(pIY)
    cv2.line(img_copy,(0,gap),(img.shape[1],gap),(255,0,0),10)
    #plt.imshow(img_copy)
    #plt.show()
    
    return np.argmax(pIY), img_copy

def h_project(img):
    
    h_proj = np.sum(img, axis=1)
    y = np.arange(img.shape[0])
    #plt.plot(h_proj, y)
    #plt.show()
    return h_proj
    
def img_splits(img, times):
    
    size, rem = np.divmod(img.shape[1] , times)
    splits = np.arange(0,img.shape[1], size)
    if rem > 0 :
        times += 1

    length = len(splits)
    for i, split in enumerate(splits):
        if i == length - 1:
            yield img[:,split:img.shape[1]-1]
        else:
            yield img[:,split:splits[i+1]]
          
def gap_splits(img, times, Yest, Sigma):
    splits = img_splits(img, times)
    gaps = np.empty(times+1)
    gap_size = np.empty(times+1)
    new_img = np.empty((img.shape[0],0))
    for i, split in enumerate(splits):
        gaps[i], split_img = gap_valley_img(split, Yest, Sigma)
        if i == 0:
            gap_size[i] = split_img.shape[0] / 2
        else:
            gap_size[i] = gap_size[i-1] + split_img.shape[0] #laatste gaat niet kloppe
        new_img = np.append(new_img, split_img, axis=1)
        
    #plt.imshow(new_img)
    #plt.show(new_img)
    
    return gaps, gap_size, new_img


def interpolate(img, gaps, gap_size):
    f2 = interp1d(gap_size, gaps, kind='cubic')
    plt.plot(gap_size, gaps, '-', gap_size, f2(gap_size), '--')
    #plt.axis((0, img.shape[1], 0, img.shape[0]))
    plt.show()
    
    
def interpolate2(img, gaps, gap_size):
    #tck = interpolate.splrep(gap_size, gaps, s=0)
    ynew = interpolate.splev(gap_size, gaps, der=0)
    plt.plot(gap_size, gaps, 'x', gap_size, ynew, '--')
    #plt.axis((0, img.shape[1], 0, img.shape[0]))
    plt.show()


# In[ ]:
if __name__ == "__main__":
    #main

    dir_radiographs = "_Data\Radiographs\*.tif"
    radiographs = load_files(dir_radiographs)

    dir_segmentations = "_Data\Segmentations\*.png"
    segmentations = load_files(dir_segmentations)

    all_landmarks = load_landmarks()
    show_teeth_points(all_landmarks[0])

    all_landmarks_std = total_procrustes_analysis(all_landmarks)
    show_teeth_points(all_landmarks_std[0])

    pca = PCA_analysis(all_landmarks_std[:,0], 8)
    show_PCAs(pca)

    pca = PCA_analysis(all_landmarks_std[:,1], 8)
    show_PCAs(pca)
    #==============================================

    plt.imshow(radiographs[0])
    plt.show()

    re = radiograph_preprocess(radiographs[0])
    plt.imshow(re)
    plt.show()

    ree = radiograph_preprocess2(radiographs[0])
    plt.imshow(ree)
    plt.show()

    gap_detection(ree)

    img = radiographs[0]
    gaps,gap_size, new_img = gap_splits(img, 20, 900, 400)
    plt.imshow(new_img)
    plt.show()


#interpolate2(img, gaps, gap_size)

