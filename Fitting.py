import cv2
import ActiveShapeModel
import numpy as np



def showImages(image,model):
	cv2.namedWindow( "Radiograph", cv2.WINDOW_AUTOSIZE )
	cv2.imshow("Radiograph",image)
	pasted = 0
	cv2.setMouseCallback('Radiograph',mousePosition,(resized_image,model))
	cv2.waitKey(0)

def reloadImage(image):
	cv2.imshow("Radiograph",image)


def mousePosition(event,x,y,flags,param):
	global pasted
	if pasted==True:
		if event == cv2.EVENT_LBUTTONDBLCLK:
			reloadImage(param[0])
			pasted=False
			return

	if pasted==False:
		reloadImage(param[0])
		image = param[0].copy()
		if event == cv2.EVENT_MOUSEMOVE:
			# print (x,y)
			# cv2.circle(image,(x,y),40,(255,0,0))
		    cropy = param[0].shape[0] - y
		    cropx = param[0].shape[1] - x
		    image[y:y+param[1].shape[0],x:x+param[1].shape[1]] = param[1][0:cropy,0:cropx]
		    cv2.imshow('Radiograph',image)
		    # param = (x,y)

		if event == cv2.EVENT_LBUTTONDBLCLK:
			print(x,y)
			print("Placing model")
			# cv2.circle(param[0],(x,y),40,(255,0,0))
			cropy = param[0].shape[0] - y
			cropx = param[0].shape[1] - x
			image[y:y+param[1].shape[0],x:x+param[1].shape[1]] = param[1][0:cropy,0:cropx]
			cv2.imshow('Radiograph',image)
			pasted=True

def moveTeeth(event,x,y,flags,param):
	global pasted
	tooth_size = param[2] # (Width, Height)
	image_center =  param[3]# (X,Y)
	top_bottom_separation = param[5] # space between top and bottom incisors
	tooth_gap = param[4]
	landmarks = param[1]
	backdrop = param[0]

	if pasted==True:
		if event == cv2.EVENT_LBUTTONDBLCLK:
			reloadImage(backdrop)
			pasted=False
			return

	if pasted==False:
		reloadImage(backdrop)
		image = backdrop.copy()
		if event == cv2.EVENT_MOUSEMOVE:
			# print (x,y)
			# cv2.circle(image,(x,y),40,(255,0,0))
		    drawTeeth(landmarks, image, tooth_size, (x,y), tooth_gap, top_bottom_separation)
		    # param = (x,y)

		if event == cv2.EVENT_LBUTTONDBLCLK:
			image = backdrop.copy()
			drawTeeth(landmarks, image, tooth_size, (x,y), tooth_gap, top_bottom_separation)
			pasted=True
		# 	print(x,y)
		# 	print("Placing model")
		# 	# cv2.circle(param[0],(x,y),40,(255,0,0))
		# 	cropy = param[0].shape[0] - y
		# 	cropx = param[0].shape[1] - x
		# 	image[y:y+param[1].shape[0],x:x+param[1].shape[1]] = param[1][0:cropy,0:cropx]
		# 	cv2.imshow('Radiograph',image)
		# 	pasted=True

def drawTeeth(landmarks,backdrop,tooth_size,image_center,tooth_gap,top_bottom_separation):
	for j in range(0,4):
			for i in range(0,40):
					x = int(landmarks[0][j][i][0]*tooth_size[0]+image_center[0]+tooth_gap*j)
					y = int(landmarks[0][j][i][1]*tooth_size[1]+image_center[1])
					# print(x)
					# print(y)
					cv2.circle(backdrop,(x,y),1,(255,255,255),-1)
					cv2.imshow("Radiograph",backdrop)

	for j in range(4,8):
		for i in range(0,40):
				x = int(landmarks[0][j][i][0]*tooth_size[0]+image_center[0]+tooth_gap*(j-4))
				y = int(landmarks[0][j][i][1]*tooth_size[1]+image_center[1]+top_bottom_separation)
				# print(x)
				# print(y)
				cv2.circle(backdrop,(x,y),1,(255,255,255),-1)
				cv2.imshow("Radiograph",backdrop)


def InitializeASM():
	dir_radiographs = "_Data\Radiographs\*.tif"
	radiographs = ActiveShapeModel.load_files(dir_radiographs)
	dir_segmentations = "_Data\Segmentations\*.png"
	segmentations = ActiveShapeModel.load_files(dir_segmentations)

	all_landmarks = ActiveShapeModel.load_landmarks()
	# show_teeth_points(all_landmarks[0])
	all_landmarks_std = ActiveShapeModel.total_procrustes_analysis(all_landmarks)
	# show_teeth_points(all_landmarks_std[0])
	# pca = PCA_analysis(all_landmarks_std[0], 8)

	img = cv2.imread("_Data/Radiographs/01.tif")
	resized_image = cv2.resize(img, (800, 400)) 
	cv2.namedWindow( "Radiograph", cv2.WINDOW_AUTOSIZE )
	cv2.imshow("Radiograph",resized_image)
	pasted = 0
	tooth_size = (200,100) # (Width, Height)
	image_center = (800/2,400/2) # (X,Y)
	top_bottom_separation = 100 # space between top and bottom incisors
	tooth_gap = 40 # space between teeth on same row
	cv2.setMouseCallback('Radiograph',moveTeeth,(resized_image,all_landmarks_std,tooth_size,image_center,tooth_gap,top_bottom_separation))
	cv2.waitKey(0)
	

	loop=1

	while loop:
		backdrop = resized_image.copy()
		cv2.imshow("Radiograph",backdrop)
		drawTeeth(all_landmarks_std,backdrop,tooth_size,image_center,tooth_gap,top_bottom_separation)

		k = cv2.waitKey(1)
		if k == ord('a'):
			loop=0
		elif k == ord('r'):
   			tooth_gap-=5
		elif k == ord('t'):
   			tooth_gap+=5
		elif k == ord('f'):
   			tooth_size = (tooth_size[0]+10,tooth_size[1]+5)
		elif k == ord('s'):
   			tooth_size = (tooth_size[0]-10,tooth_size[1]-5)

	

	

	
	# cv2.setMouseCallback('Radiograph',mousePosition,(resized_image,model))
	cv2.waitKey(0)
	
if __name__ == "__main__":
	pasted = False
	# img = cv2.imread("_Data/Radiographs/01.tif")
	# resized_image = cv2.resize(img, (800, 400)) 
	# model = cv2.imread("_Data/Radiographs/02.tif")
	# resized_model = cv2.resize(model,(200,100))
	# showImages(resized_image,resized_model)
	InitializeASM()

