from tkinter import *
import ActiveShapeModel


def greet():
	print("test")

dir_radiographs = "_Data\Radiographs\*.tif"
radiographs = ActiveShapeModel.load_files(dir_radiographs)
dir_segmentations = "_Data\Segmentations\*.png"
segmentations = ActiveShapeModel.load_files(dir_segmentations)
all_landmarks = ActiveShapeModel.load_landmarks()
# show_teeth_points(all_landmarks[0])
all_landmarks_std = ActiveShapeModel.total_procrustes_analysis(all_landmarks)
# show_teeth_points(all_landmarks_std[0])
pca = ActiveShapeModel.PCA_analysis(all_landmarks_std[:,0], 8)

root = Tk()

Title = Label(root, text="Toolbox")
Title.pack()

seperator = Label(root, text="===================")
seperator.pack()

showModelButton = Button(root, text="Show teeth points", command=lambda:ActiveShapeModel.show_teeth_points(all_landmarks[0]))
showModelButton.pack()

toothGapLabel = Label(root,text="Tooth Gap: ")
toothGapLabel.pack(side=LEFT)
toothGapButton = Button(root, text="-", command=greet,height = 1, width = 5)
toothGapButton.pack(side=LEFT)
toothGapButton = Button(root, text="+", command=greet,height = 1, width = 5)
toothGapButton.pack()



root.mainloop()