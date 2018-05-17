
# coding: utf-8

# In[ ]:


import numpy as np
import matplotlib.pyplot as plt
from skimage.color import rgb2gray
from skimage.filters import gaussian
import ActiveShapeModel 

def calc_internal(p1,p2):
    return np.sum( (p2 - p1)**2 )

def calc_external_img(img):
    
    img = rgb2gray(img)

    sobelx = cv.Sobel(img,cv.CV_64F,1,0,ksize=5)
    sobely = cv.Sobel(img,cv.CV_64F,0,1,ksize=5)

    return -(sobelx**2 + sobely**2)
    
def calc_external(p, external_img):
    
    p = np.around(p)
    p = p.astype(int)
    
    return external_img[p[0],p[1]]
    
def calc_energy(p1, p2, external_img, alpha):
     
    internal = calc_internal(p1,p2)
    external = calc_external(p1, external_img)
    
    return np.sum(internal) + alpha * external


def get_point_state(point, number):
    
    n=0
    for i in range(-2,3):
        for j in range(-2,3):          
            if n == number:
                return np.array([point[0]+i , point[1]+j])
            n +=1
    return

def unpack(number, back_pointers, points):
    
    size = len(points)
    new_points = np.empty((size,2))
    
    new_points[-1] = get_point_state(points[-1],number)
    pointer = back_pointers[-1,number]
    
    for i in range(size-2, -1, -1):
        
        new_points[i] = get_point_state(points[i],pointer)
        pointer = back_pointers[i,pointer]
        
    return new_points

#https://courses.engr.illinois.edu/cs447/fa2017/Slides/Lecture07.pdf
def viterbi(points, img, alpha):
    size = len(points)
    num_states = 25
    trellis = np.empty((size, num_states), dtype=np.float32)
    back_pointers = np.empty((size, num_states), dtype=int)
    external_img = calc_external_img(img)
    
    #init
    trellis[0,:] = np.zeros(num_states)
    back_pointers[0,:] = np.zeros(num_states)
    
    #recursion
    for i in range(1, size):
        for t in range(num_states):
            trellis[i,t] = np.inf
            for d in range(num_states):
                p1 = get_point_state(points[i-1], d)
                p2 = get_point_state(points[i],t)
                energy_trans = calc_energy(p1, p2, external_img, alpha)
                energy_trans = int(energy_trans)
                
                tmp = trellis[i-1,d] + energy_trans
                
                if(tmp < trellis[i,t]):
                    trellis[i,t] = tmp
                    back_pointers[i,t] = d
            
    #find best
    t_best, vit_min = 0, np.inf
    for t in range(num_states):
        if(trellis[size-1, t] < vit_min):
            t_best = t
            vit_min = trellis[size-1, t]

    return unpack(t_best, back_pointers, points)

def active_contour(points, img, alpha, max_loop):
    
    old_points = points
    for i in range(max_loop):
        new_points = viterbi(old_points, img, alpha)
        if np.array_equal(new_points, old_points):
            print(i)
            break
            
        #old_points = new_points
        head, tail = np.split(new_points, [1])
        old_points = np.append(tail, head).reshape(new_points.shape)
        
    return new_points


# In[ ]:


if __name__ == "__main__":

