import numpy as np
import cv2
from pybrain.tools.customxml.networkreader import NetworkReader
from fractions import gcd

windowWidth = 28
windowHeight = 32
slidingStep = 5

factor = gcd(windowWidth,windowHeight)
expandStepH = windowHeight/factor
expandStepW = windowWidth/factor

imgPath = raw_input('img to be read: ')
img = cv2.imread(imgPath,0)
h,w = img.shape
posList = []

netRead = NetworkReader.readFrom('trainWithMomen0.0WeiDecay0.003MaxE1000ExamRefreshedNormalized.xml') 

maxK = min( (w - windowWidth)/expandStepW , (h - windowHeight)/expandStepH)
print 'maxk: ' + str(maxK)
print 'jmax: '+str((h - windowHeight)/slidingStep)+' imax: '+str((w - windowWidth)/slidingStep)
test = raw_input('go?')


if w >= 28 and h >= 32 and test is 'y':
	for k in range(0,maxK): # expand window by step
		windowWidth = windowWidth + 2 * k * expandStepW
		windowHeight = windowHeight + 2 * k * expandStepH
		slidingStep = (windowWidth + windowHeight) / 10

		for i in range(0,(w - windowWidth)/slidingStep + 1):
			for j in range(0,(h - windowHeight)/slidingStep + 1):# sliding window with current window size
				windowImg = (img[j*slidingStep: j*slidingStep + windowHeight, i*slidingStep: i*slidingStep + windowWidth])
				if k > 0:
					windowImg = cv2.resize(windowImg,(28,32),interpolation = cv2.INTER_AREA) 

				windowImg = windowImg.ravel() / 127.5 - 1

				
				if netRead.activate(windowImg) > 0.8 : 
					print netRead.activate(windowImg)
					posList.append((i*slidingStep,j*slidingStep,windowWidth,windowHeight))
					print 'found (%s,%s,%s,%s)' %(i*slidingStep,j*slidingStep,windowWidth,windowHeight)
					print 'k:%s,i:%s,j%s\n' %(k,i,j)
					

else : raw_input('image input too small')

img = cv2.imread(imgPath)

for (i,j,wW,wH) in posList: 
	img = cv2.rectangle(img,(i,j),(i+wW,j+wH),(255,0,0),1)

cv2.namedWindow('img',cv2.WINDOW_NORMAL)
cv2.imshow('img',img)
cv2.waitKey(0)
cv2.destroyAllWindows()