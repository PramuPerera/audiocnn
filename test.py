import scipy.io
import tensorflow as tf
import numpy
import numpy as np
import json
import cv2
import librosa
import keras
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras.layers.normalization import BatchNormalization
from keras.optimizers import Adam
from keras.models import model_from_json
#import keras_resnet.models
from keras.models import Model
import pylab
from sklearn import svm
import sklearn.metrics
from sklearn.manifold import TSNE
from ggplot import *
import pandas as pd
from pandas import Timestamp
import matplotlib.pyplot as plt



def test(filename, trainname, outname):
	scores = scipy.io.loadmat(trainname)
	score = scores['score']
	lbl = scores['class']
	lbls =[]
	n_sne = 7000
	tsne = TSNE(n_components=2, verbose=1, perplexity=40, n_iter=300)

	for i in lbl:
		if i[3]==1:
			lbls.append(1)
		else:
			lbls.append(0)
	trainset = score[0:int(len(score)*0.5)][:]
	trainlbl = lbls[0:int(len(score)*0.5)][:]
	posset = [trainset[i][:] for i,j in enumerate(trainlbl) if j==1]
	testset = score[int(len(score)*0.5):-1][:]
	testlbl = lbls[int(len(score)*0.5):-1][:]
	acc = []
	nus = [0.0001, 0.001, 0.01, 0.1, 0.3, 0.5, 0.7, 0.9 ]
	gammas = [0.001, 0.005, 0.01, 0.05, 0.1, 1, 10 ,100 ,1000]
	for nu in  nus:
	    for gamma in gammas:
		clf = svm.OneClassSVM( nu= nu, kernel="rbf", gamma=gamma)
		clf.fit(posset)
		y_pred_train = clf.predict(testset)
		y_pred_train[y_pred_train==-1] = 0 
		acc.append(sklearn.metrics.accuracy_score(testlbl, y_pred_train))
	print(acc)
	print(numpy.max(acc))
	nu = nus[numpy.argmax(acc)/9]
	gamma = gammas[numpy.argmax(acc)-numpy.argmax(acc)/9*9]


	clf = svm.OneClassSVM( nu= nu, kernel="rbf", gamma=gamma)
	clf.fit(posset)
	y_pred_train = clf.predict(testset)
	y_pred_train[y_pred_train==-1] = 0
	print(sklearn.metrics.accuracy_score(testlbl, y_pred_train))
	#print([nu, gamma])

	# Test
	scores = scipy.io.loadmat(filename)
	score = scores['score']
	lbl = scores['class']
	lbls =[]
	c = 0
	for i in lbl:
		if i[3]==1:
			lbls.append(1)
			c+=1
		else:
			lbls.append(0)
	print(float(c)/len(lbls))
	y_pred_train = clf.predict(score)
	y_pred_train[y_pred_train==-1] = 0 
	acc1 = (sklearn.metrics.balanced_accuracy_score(lbls, y_pred_train))
	acc2 = (sklearn.metrics.accuracy_score(lbls, y_pred_train))
	f1 = sklearn.metrics.f1_score(lbls, y_pred_train)
	auc = sklearn.metrics.roc_auc_score(lbls, y_pred_train)
	tsne_results = tsne.fit_transform(np.concatenate((posset, score)))
	lbls = [y+1 for y in lbls]
	col = np.concatenate((np.zeros(np.shape(posset)[0]),lbls))
	xs = tsne_results[:,0]
	ys = tsne_results[:,1]
	str = ['Training', 'Testing abormal', 'Testing normal']
	for i in range(3):
    		xi = [xs[j] for j  in range(len(xs)) if col[j] == i]
    		yi = [ys[j] for j  in range(len(xs)) if col[j] == i]
    		plt.scatter(xi, yi, label=str[i], alpha = 0.5)
	plt.legend()
	plt.savefig(outname)
	plt.gcf().clear()
	#chart = ggplot( df_tsne, aes(x='x-tsne', y='y-tsne', color='label') ) \
        #	+ geom_point(size=70,alpha=0.1) \
        # 	+ ggtitle("tSNE dimensions colored by digit")
	#chart

	# argmax of the layer
	#y_pred_train = numpy.argmax(score,1)
	#y_pred_train = [int(i==3) for i in y_pred_train]
	#print(y_pred_train)
	#acc2 = (sklearn.metrics.balanced_accuracy_score(lbls, y_pred_train))
	#f2 = sklearn.metrics.f1_score(lbls, y_pred_train)
	return([acc1, f1, acc2, auc])
