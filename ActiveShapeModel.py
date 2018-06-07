
# coding: utf-8

# In[94]:


import MatchingModelPoints as match
import FitFunction as fit
import FileManager as fm
import numpy as np
import matplotlib.pyplot as plt
import Image_preperation as prep
import PCA_analysis as PCA

def active_shape(edge_img, tooth_points, pca_tooth, length):

    new_points, error = fit.fit_measure(tooth_points, length, edge_img)
    b, pose_param = match.match_model_points(new_points, pca_tooth)

    x = match.generate_model_point(b, pca)
    return match.inv_transform(x.reshape(40,2),pose_param)

def active_shape_n_times(edge_img, tooth_points, pca_tooth, length, n_times):
    
    points = [0] * (n_times+1)
    points[0] = tooth_points

    for i in range(n_times):
        points[i+1] = active_shape(edge_img, points[i], pca_tooth, length)
                   
    return points

def preperation(radiograph, tooth_variations):
    
    median = prep.median_filter(radiograph)
    edge_img = prep.edge_detection_low(median)
    pca_tooth = PCA.PCA_analysis(tooth_variations, None)
    
    return edge_img, pca_tooth


# In[92]:


if __name__ == "__main__":
    
    teeth = np.load('initial_position.npy')
    tooth = teeth[0,0]
    tooth = tooth/0.3

    radiographs = fm.load_radiographs()
    radiograph = radiographs[0]

    landmarks = fm.load_landmarks_std()
    tooth_variations = landmarks[:,0]

    edge_img, pca_tooth = preperation(radiograph, tooth_variations)

    fig, ax = plt.subplots(figsize=(15, 15))
    plt.imshow(radiograph)
    plt.plot(tooth[:,0], tooth[:,1], 'ro', markersize=1)
    plt.show()


# In[95]:


points = active_shape_n_times(edge_img, tooth, pca_tooth, 5, 10)


# In[102]:


new_points = points[1]


# In[103]:


fig, ax = plt.subplots(figsize=(15, 15))
plt.imshow(radiograph)
plt.plot(new_points[:,0], new_points[:,1], 'ro', markersize=1)
plt.show()

