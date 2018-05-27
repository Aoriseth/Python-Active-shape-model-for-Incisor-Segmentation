
# coding: utf-8

# In[7]:


import Image_preperation as prep
import numpy as np
import matplotlib.pyplot as plt
import math

def fit_measure(points, length, edge_img):
    
    size = len(points)
    new_points = np.empty(size,2)
    total_error = 0
    
    for i in range(size):
        if(i==size-1):
            p1, p2, p3 = points[i-1], points[i], points[0] 
        else:
            p1, p2, p3 = points[i-1], points[i], points[i+1]

        p2_new = strongest_edge_point_on_normal(p1, p2, p3 ,length, edge_img)
        total_error += error_measure(p2, pnew)
        new_points[i] = p2_new
        
    return total_error;   
        
def error_measure(p1, p2):
    
    x1, y1 = p1
    x2, y2 = p2
    #dist = sqrt( (x2 - x1)**2 + (y2 - y1)**2 )
    return math.hypot(x2 - x1, y2 - y1)
    
         
def strongest_edge_point_on_normal(a,b,c,length, edge_img):
    
    rad = get_normal_angle(a,b,c)
    points = get_points_on_angle(b, rad, length)
    edge_strength = edge_strength_at_points(points, edge_img)
    id_edge_point = np.argmax(edge_strength)
    edge_point = points[id_edge_point]

    return edge_point

def edge_strength_on_normal(a,b,c,length, edge_img):
    
    rad = get_normal_angle(a,b,c)
    points = get_points_on_angle(b, rad, length)
    return edge_strength_at_points(points, edge_img)

def get_normal_angle(a,b,c):
    
    b_proj = project_on(b, a,c)
    b_norm = np.add(b, [2,0])
    return calc_angle(b_proj, b, b_norm)

def project_on(x, a,c):
    n = np.subtract(a, c)
    n = np.divide(n, np.linalg.norm(n, 2))

    return c + n*np.dot(np.subtract(x , c), n)

def calc_angle(a,b,c):
    
    ba = np.subtract(a, b)
    bc = np.subtract(c , b)

    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    return np.arccos(cosine_angle)


def get_y_point(x,rad):
    return int(np.around(math.tan(rad) * x))

def get_points_on_angle(point, rad, length):
    
    if  rad > math.pi/4:
        switched = True
        new_rad = math.pi/4 - ( rad - math.pi/4 )
    else:
        switched = False
        new_rad = rad
    
    points = np.empty((2*length+1, 2))
    for i, x in enumerate(range(-length, length+1)):
        
        y = get_y_point(x, new_rad)

        if switched:
            sub_point = [y,x] 
        else:
            sub_point = [x,y]
            
        points[i] = np.subtract(point, sub_point)
        
    return points

def edge_strength_at_points(points ,edge_img):
    
    gradient = np.empty(len(points))
    for i, p in enumerate(points):
        gradient[i] = edge_img[int(p[0]),int(p[1])]
        
    return gradient

def normalize(x):
    return x / np.linalg.norm(x)


def load_tooth():
    init = np.load("initial_position.npy")
    return init[0,4,:,:]/0.3   

def show_with_points(img, points):
    fig, ax = plt.subplots(figsize=(7, 7))
    plt.imshow(img)
    plt.plot(points[:,0], points[:,1], 'ro', markersize=2)
    plt.show()


# In[8]:


if __name__ == "__main__":

    img = prep.load_image()
    piece = img[700:1300,1200:1800]

    tooth = load_tooth()

    tooth2 = tooth
    tooth2[:,0]=tooth[:,0]-1200
    tooth2[:,1]=tooth[:,1]-700

    points = tooth2[10:13]

    show_with_points(piece, points)


    piece = prep.median_filter(piece)
    edge_img = prep.edge_detection_low(piece)
    show_with_points(edge_img, points)
    a,b,c = points
    edges = edge_strength_on_normal(a,b,c,20, edge_img)


    y = np.arange(-20,21)
    plt.plot(y, normalize(edges))
    plt.show()


# In[2]:


len([[2,2],[1,1]])

