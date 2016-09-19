from os import listdir, makedirs
from os.path import exists
import numpy as np
import cv2
from pybrain.tools.shortcuts import buildNetwork
from pybrain.datasets import ClassificationDataSet
from pybrain.supervised.trainers import BackpropTrainer
from pybrain.tools.customxml.networkwriter import NetworkWriter
from pybrain.tools.customxml.networkreader import NetworkReader

xmlNames = []
#xmlNames.append('trainedWithMomen0WeiDecay0MaxE1000.xml')
#xmlNames.append('trainWithMomen0.1WeiDecay0.01MaxE800.xml')
#xmlNames.append('trainWithMomen0.1WeiDecay0.01MaxE1000.xml')
#xmlNames.append('trainWithMomen0.1WeiDecay0.1MaxE1000.xml')
#xmlNames.append('trainWithMomen0.3WeiDecay0.01MaxE1000.xml')
#xmlNames.append('trainWithMomen0.5WeiDecay0.01MaxE1000.xml')
#xmlNames.append('trainWithMomen0.4WeiDecay0.01MaxE1000.xml')
#xmlNames.append('trainWithMomen0.35WeiDecay0.01MaxE1000.xml')
#xmlNames.append('trainWithMomen0.25WeiDecay0.01MaxE1000.xml')
#xmlNames.append('trainWithMomen0.325WeiDecay0.01MaxE1000.xml')
#xmlNames.append('trainWithMomen0.31WeiDecay0.01MaxE1000.xml')
#xmlNames.append('trainWithMomen0.3WeiDecay0.1MaxE1000.xml')
#xmlNames.append('trainWithMomen0.3WeiDecay0.03MaxE1000.xml')
#xmlNames.append('trainWithMomen0.3WeiDecay0.015MaxE1000.xml')
#xmlNames.append('trainWithMomen0.3WeiDecay0.005MaxE1000.xml')
#xmlNames.append('trainWithMomen0.3WeiDecay0.0075MaxE1000.xml')
#xmlNames.append('trainWithMomen0.3WeiDecay0.01MaxE1000Exam2000.xml')
#xmlNames.append('trainWithMomen0.1WeiDecay0.01MaxE1000Exam2000.xml')
#xmlNames.append('trainWithMomen0.0WeiDecay0.01MaxE1000Exam2000.xml')
#xmlNames.append('trainWithMomen0.0WeiDecay0.03MaxE1000Exam2000.xml')
#xmlNames.append('trainWithMomen0.0WeiDecay0.0MaxE1000Exam2000.xml')
#xmlNames.append('trainWithMomen0.3WeiDecay0.005MaxE1000Exam2000.xml')
#xmlNames.append('trainWithMomen0.0WeiDecay0.0MaxE1000Exam2000Normalized.xml')
#xmlNames.append('trainWithMomen0.3WeiDecay0.01MaxE1000Exam2000Normalized.xml')
#xmlNames.append('trainWithMomen0.0WeiDecay0.0MaxE1000Exam2400Normalized.xml')
#xmlNames.append('trainWithMomen0.0WeiDecay0.01MaxE1000Exam2400Normalized.xml')
#xmlNames.append('trainWithMomen0.1WeiDecay0.0MaxE1000Exam2400Normalized.xml')
#xmlNames.append('trainWithMomen0.0WeiDecay0.003MaxE1000Exam2400Normalized.xml')
#xmlNames.append('trainWithMomen0.0WeiDecay0.003MaxE1000Exam6161Normalized.xml')
xmlNames.append('trainWithMomen0.0WeiDecay0.003MaxE1000ExamRefreshedNormalized.xml')

for xmlName in xmlNames: 
	valiDir = 'validation/'
	filenames = [f for f in listdir(valiDir)]
	netRead = NetworkReader.readFrom(xmlName) 
	#raw_input(netRead)
	totalNum = len(filenames)
	truePositives = 0
	trueNegatives = 0
	falsePositives = 0
	falseNegatives = 0

	for f in filenames: 
		img = cv2.imread(valiDir + f , 0).ravel()
		img = img / 127.5 - 1
		output = netRead.activate(img)
		if output >= 0.5 and f.find('yale') >= 0 : 
			truePositives = truePositives + 1
		if output < 0.5 and f.find('yale') < 0:
			trueNegatives = trueNegatives + 1
		if output >= 0.5 and f.find('yale') < 0:
			falsePositives = falsePositives + 1
		if output < 0.5 and f.find('yale') >= 0:
			falseNegatives = falseNegatives + 1

		#print netRead.activate(img.ravel()),'\n'
		if f.find('Me') >= 0:
			cv2.namedWindow(f,cv2.WINDOW_NORMAL)
			print output
			cv2.imshow(f,cv2.imread(valiDir + f,0))
			cv2.waitKey(0)
			cv2.destroyWindow(f)
			

	# correctly detected face by detected as face
	precision = float(truePositives) / (truePositives + falsePositives) 
	# correctly detected face by real faces
	recall = float(truePositives) / (truePositives + falseNegatives)
	accuracy = float(truePositives+trueNegatives) / totalNum
	f1Score = 2 * precision * recall / (precision + recall)

	print'\n%s' %(xmlName)
	print 'truePositives %s + falseNegatives %s == num of example faces %s' %(truePositives,falseNegatives,truePositives+falseNegatives)
	print 'trueNegatives %s + falsePositives %s == num of example nonFaces %s' %(trueNegatives,falsePositives,trueNegatives+falsePositives)
	print 'trueDetection %s   falseDetection %s' %(truePositives+trueNegatives,falsePositives+falseNegatives)
	print 'accuracy: %s%%' %(accuracy * 100)
	print 'precision (correctly detected face by detected as face): %s%%' %(precision * 100)
	print 'precision (correctly detected face by real faces): %s%%' %(recall * 100)
	print 'f1-score: %s' %(f1Score)

	raw_input('done with deprecated examples removed\n-----------------------------------------------------------------------\n')


