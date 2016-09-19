from os import listdir, makedirs
from os.path import exists
import numpy as np
import cv2
from pybrain.tools.shortcuts import buildNetwork
from pybrain.datasets import ClassificationDataSet
from pybrain.supervised.trainers import BackpropTrainer
from pybrain.tools.customxml.networkwriter import NetworkWriter
from pybrain.tools.customxml.networkreader import NetworkReader
from pybrain.structure import SigmoidLayer

DS = ClassificationDataSet(896,class_labels = ['notFace','Face'])

posDir = 'pos/'
posFilenames = [f for f in listdir(posDir)]
for f in posFilenames: 
	img = (cv2.imread(posDir + f, 0)).ravel()
	img = img / 127.5 - 1
	DS.appendLinked(img,[1])

negDir = 'neg/'
negFilenames = [f for f in listdir(negDir)]
for f in negFilenames: 
	img = cv2.imread(negDir + f, 0).ravel()
	img = img / 127.5 - 1
	DS.appendLinked(img,[0])# Dataset setup here

Momen = 0.0
WeiDecay = 0.003
print 'training...'
net = buildNetwork(896,100,10,1, bias = True,outclass = SigmoidLayer)
trainer = BackpropTrainer(net, DS,momentum=Momen, weightdecay=WeiDecay)
proportion2Cost = trainer.trainUntilConvergence(validationProportion = 0.20, maxEpochs = 1000, continueEpochs = 10)
raw_input(proportion2Cost)

xmlName = 'trainWithMomen%sWeiDecay%sMaxE1000ExamRefreshedNormalized.xml'%(Momen,WeiDecay)

if not exists(xmlName):
	NetworkWriter.writeToFile(net, xmlName) 
	print 'output xml'
else :
	print 'xml name already existed'
