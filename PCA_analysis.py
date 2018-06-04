
# coding: utf-8

# In[5]:


import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
import FileManager


def PCA_analysis(data, number_of_components):
    
    data = data.reshape(-1,80)
    pca = PCA(n_components= number_of_components)
    pca.fit(data)
    
    return pca

def PCA_reconstruction(pca, data):
    
    projections = pca.transform(data)
    reconstructions = pca.inverse_transform(projections)
    return reconstructions

def get_eigenvalues(pca):
    
    return pca.explained_variance_

def get_eigenvectors(pca):
    return pca.components_

def get_mean(pca):
    return pca.mean_


######### Visualization #########


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
        FileManager.show_tooth_points(vector, False)
     
    plt.show()


# In[6]:


if __name__ == "__main__":
    #main
    all_landmarks_std = FileManager.load_landmarks_std()

    pca = PCA_analysis(all_landmarks_std[:,0], None)
    show_PCAs(pca)

