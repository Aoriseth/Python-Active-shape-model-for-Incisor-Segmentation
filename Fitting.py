import cv2

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

	


pasted = False
img = cv2.imread("_Data/Radiographs/01.tif")
resized_image = cv2.resize(img, (800, 400)) 
model = cv2.imread("_Data/Radiographs/02.tif")
resized_model = cv2.resize(model,(200,100))
showImages(resized_image,resized_model)
