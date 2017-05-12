import glob
import os
import numpy as np
from skimage.io import imread
from skimage.transform import resize


def getFolderNames(root):
	foldernames = os.listdir(root)
	return foldernames

def getImagesFromFolder(root,extensions=['png','jpg']):
	filenames = glob.glob(root+'*')
	filenames = [x for x in filenames if x.split('.')[-1] in extensions]
	imgSet = [imread(filename) for filename in filenames]
	return imgSet

def getImgDictionary(root):
	foldernames = getFolderNames(root)
	imgDict = {}
	for foldername in foldernames:
		imgSet = getImagesFromFolder(root+foldername+'/',['png'])
		# print(foldername+'  :  '+str(len(imgSet))+' images')
		imgDict[foldername] = imgSet
	return imgDict

def resizeImages(imgDict,shape):
	for label in imgDict.keys():
		imgSet = imgDict[label]
		imgSet = [resize(img,shape,mode='reflect') for img in imgSet]
		imgDict[label] = imgSet
	return imgDict

def prepNumpyArrays(imgDict):
	X = []
	Y = []
	for label in imgDict.keys():
		imgSet = imgDict[label]
		X += imgSet
		Y += [label for x in range(len(imgSet))]
	X = np.array(X)
	Y = np.array(Y)
	return X,Y


print('TRAINING SET')
root = 'train/'
imgDict = getImgDictionary(root)
imgDict = resizeImages(imgDict,(32,32))
trainX,trainY = prepNumpyArrays(imgDict)
print(trainX.shape)
print(trainY.shape)

# print('TEST SET')
# root = 'test/'
# getImgDictionary(root)