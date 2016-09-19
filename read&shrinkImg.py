from os import listdir, makedirs
from os.path import exists
import numpy as np
import cv2


posDir = 'pos/'
if not exists(posDir):
	makedirs(posDir)

folders = range(1,41)
#folders.pop(13) # yaleB14 missing



for i in  folders: 
	Dir = 'ATT/s%s/' %(i)
	filenames = [f for f in listdir(Dir) if f.endswith('.pgm')]
	#filenames.pop() # get rid of the ambdient one

	for f in filenames:
		img = cv2.imread(Dir + f ,0)
		img = img[16:112,4:88]
		img = cv2.resize(img,(28,32),interpolation = cv2.INTER_AREA) 
		cv2.imwrite(posDir + 'yaleATTs%s'%i + f ,img)
		

	#print 'yaleB%02d done' %(i) 
	print 'yaleATTs%s done' %(i)

