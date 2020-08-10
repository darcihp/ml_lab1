#!/usr/bin/env python
# -*- encoding: iso-8859-1 -*-
import sys
import numpy
import rospy
import roslib
from sklearn.model_selection import train_test_split
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.datasets import load_svmlight_file
from sklearn import preprocessing
import pylab as pl
import rospkg
import os
from sklearn.metrics import precision_recall_fscore_support
import datetime
import matplotlib.pyplot as plt
import itertools

class_names = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]

def plot_confusion_matrix(cm, classes, normalize=False, title='Confusion matrix', cmap=plt.cm.Blues):
	plt.imshow(cm, interpolation='nearest', cmap=cmap)
	plt.title(title)
	plt.colorbar()
	tick_marks = numpy.arange(len(classes))
	plt.xticks(tick_marks, classes, rotation=45)
	plt.yticks(tick_marks, classes)

	if normalize:
		cm = cm.astype('float') / cm.sum(axis=1)[:, numpy.newaxis]
		print("Normalized confusion matrix")
	else:
		print('Confusion matrix, without normalization')

	print(cm)

	thresh = cm.max() / 2.
	for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        	plt.text(j, i, "{:0.2f}".format(cm[i, j]), horizontalalignment="center", color="white" if cm[i, j] > thresh else "black")

	plt.tight_layout()
	plt.ylabel('True label')
	plt.xlabel('Predicted label')

def lda(data_train, data_test, _fout, _ftime):

	# Loads data
        print ("Loading data...")
        X_train, y_train = load_svmlight_file(data_train)
        X_test, y_test = load_svmlight_file(data_test)

        X_train = X_train.toarray()
        X_test = X_test.toarray()

	print("Normalizing...")
        # Fazer a normalizacao dos dados #######
        scaler = preprocessing.MinMaxScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.fit_transform(X_test)

	for _train_samples in range(1000, 21000, 1000):
		print("Train samples: %d" %_train_samples)
		print("Creating LDA...")
        	# Cria um LDA
		lda = LinearDiscriminantAnalysis()

	        print ('Fitting LDA...')
		lda.fit(X_train[0:_train_samples], y_train[0:_train_samples])

	        print ('Predicting...')
		#Mensura tempo - Start
		start = datetime.datetime.now()

	        # Predicao do classificador
	        y_pred = lda.predict(X_test)

		#Mensura tempo - Stop
		stop = datetime.datetime.now()

		#Delta de tempo
		delta = stop - start

		_ftime.write("lda_" + str(_train_samples) + ". Start: " + str(start) + ". Stop: " + str(stop) +". Delta: " + str(delta) )
		_ftime.write("\n")

		# Determina acurácia
		accuracy = lda.score(X_test, y_test)
	        # cria a matriz de confusao
	        cm = confusion_matrix(y_test, y_pred)

		numpy.set_printoptions(precision=2)

		plt.figure()
		plot_confusion_matrix(cm, classes=class_names, normalize=True, title="Linear Discriminant Analysis: "+str(_train_samples))
		plt.savefig(ml_lab1_path + "/lda/lda_"+ str(_train_samples)+ "_cm.png")

		print("Train Samples: %d, Accuracy: %f" %(_train_samples, accuracy))
		precision, recall, f1_score, support = precision_recall_fscore_support(y_test, y_pred, average='weighted')
		_fout.write(str(_train_samples) + " " + str(accuracy) + " " + str(precision) + " " + str(recall) + " " + str(f1_score))
        	_fout.write("\n")

def main(args):

	try:
	        rospy.init_node('n_lab_2_knn', anonymous=True)
        	rospack = rospkg.RosPack()

		#Caminho do package
		global ml_lab1_path
		ml_lab1_path = rospack.get_path("ml_lab2")
		ml_lab1_path += "/scripts"

		#Abre arquivo para escrita de resultados
		fout = open(ml_lab1_path + "/lda/results_lda", "w")
		#Abre arquivo para histórico das datas
		ftime = open(ml_lab1_path + "/lda/results_time_lda", "w")

		lda(ml_lab1_path + "/data/train.txt", ml_lab1_path + "/data/test.txt", fout, ftime)

		fout.close
		ftime.close

		print("Done")

        except KeyboardInterrupt:
	        rospy.loginfo("Shutting down")

if __name__ == "__main__":
        main(sys.argv)


