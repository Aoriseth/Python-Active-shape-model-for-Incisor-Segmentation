
# coding: utf-8

# In[2]:


import MatchingModelPoints as match
import FitFunction as fit
import FileManager as fm
import numpy as np
import matplotlib.pyplot as plt
import Image_preperation as prep
import PCA_analysis as PCA

def active_shape(img, tooth_points,tooth_variations, length, n_times):
    
    points = tooth_points
    median = prep.median_filter(img)
    edge_img = prep.edge_detection_low(median)
    pca = PCA.PCA_analysis(tooth_variations, None)

    for i in range(n_times):
    
        new_points, error = fit.fit_measure(points, length, edge_img)
        b, pose_param = match.match_model_points(new_points, pca)
        
        x = match.generate_model_point(b, pca)
        points = match.inv_transform(x.reshape(40,2),pose_param)
                   
    return points

#MatchingModelPoints.match_model_points(Y, pca)


# In[3]:


teeth = np.load('initial_position.npy')
tooth = teeth[0,0]
tooth = tooth/0.3

radiographs = fm.load_radiographs()
radiograph = radiographs[0]

landmarks = fm.load_landmarks_std()
all_tooth_variations = landmarks[:,0]

#fm.show_tooth_points(tooth)
pca = PCA.PCA_analysis(all_tooth_variations, None)

fig, ax = plt.subplots(figsize=(15, 15))
plt.imshow(radiograph)
plt.plot(tooth[:,0], tooth[:,1], 'ro', markersize=1)
plt.show()


# In[4]:


radiograph.shape


# In[4]:


points = active_shape(radiograph, tooth, all_tooth_variations, 20, 1)


# In[ ]:


fig, ax = plt.subplots(figsize=(15, 15))
plt.imshow(radiograph)
plt.plot(points[:,0], points[:,1], 'ro', markersize=1)
plt.show()

