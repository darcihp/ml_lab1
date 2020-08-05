#!/usr/bin/env python
# -*- encoding: iso-8859-1 -*-
import sys
import numpy
import rospy
import roslib
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
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

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
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
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

def knn(data, _n_neighbors, _metric, _fout, _x, _y, _arquivo):

	# loads data
        print ("Loading data...")
        X_data, y_data = load_svmlight_file(data)
        # splits data
        print ("Spliting data...")
        X_train, X_test, y_train, y_test =  train_test_split(X_data, y_data, test_size=0.5, random_state = 5)

        X_train = X_train.toarray()
        X_test = X_test.toarray()

        # fazer a normalizacao dos dados #######
        #scaler = preprocessing.MinMaxScaler()
        #X_train = scaler.fit_transform(X_train_dense)
        #X_test = scaler.fit_transform(X_test_dense)

	print("Creating KNN. n_neighbors: %d, metric: %s" %(_n_neighbors, _metric))

        # cria um kNN
	neigh = KNeighborsClassifier(n_neighbors=_n_neighbors, metric=_metric)

        print ('Fitting knn')
        neigh.fit(X_train, y_train)

        # predicao do classificador
        print ('Predicting...')
        y_pred = neigh.predict(X_test)

        # mostra o resultado do classificador na base de teste
        #print ('Accuracy: ',  neigh.score(X_test, y_test))
	accuracy = neigh.score(X_test, y_test)

        # cria a matriz de confusao
        cm = confusion_matrix(y_test, y_pred)

	numpy.set_printoptions(precision=2)

	plt.figure()
	plot_confusion_matrix(cm, classes=class_names, normalize=False, title=_arquivo+" K: "+str(_n_neighbors))
	# Plot normalized confusion matrix
	plt.savefig(_arquivo +str(_n_neighbors)+"_cm.png")


	print("X: %s, Y: %s, Accuracy: %f" %(_x, _y, accuracy))

	precision, recall, f1_score, support = precision_recall_fscore_support(y_test, y_pred, average='weighted')
	#_fout.write(str(_x) +  " " + str(_y) + " " + str(accuracy) + " " + str(precision) + " " + str(recall) + " " + str(f1_score))
        #_fout.write("\n")

def main(args):

	#_n_neighbors = [3, 5, 7, 11, 15]
	#_metrics = ["euclidean", "manhattan", "chebyshev", "minkowski"]

	_n_neighbors = [1, 15]
	_metrics = ["euclidean"]
	#_metrics = ["euclidean"]

        #if(len(sys.argv) < 3):
        #        print("./knn.py <n_neighbors[3, 5, 7, ...]> <metric[euclidean, manhattan, chebyshev, minkowski]>")
	#	return -1
	try:
	        rospy.init_node('n_lab_1_knn', anonymous=True)
        	rospack = rospkg.RosPack()

		#Caminho do package
		ml_lab1_path = rospack.get_path("ml_lab1")
		ml_lab1_path += "/scripts"

		#Percorre diretório para ler arquivos
		#for _, _, arquivos in os.walk(ml_lab1_path + "/features"): print("")

		arquivos = ["f_119-26-96", "f_179-41-96", "f_305-76-26", "f_345-86-26"]

		#Abre arquivo para histórico das datas
		ftime = open(ml_lab1_path + "/results_time", "w")

		#Percorre array de métricas
		for _metric in _metrics:
		#Percorre arrya de neighbors
			for _n_neighbor in _n_neighbors:

				start = datetime.datetime.now()

				#Abre arquivo para escrita de resultados
				fout = open(ml_lab1_path + "/results_" + str(_n_neighbor)  +"_"+ _metric, "w")

				for arquivo in arquivos:
					print(arquivo)
					aux = arquivo.split("_")[1]
					x = aux.split("-")[1]
					y = aux.split("-")[2]
					knn(ml_lab1_path + "/features/" + arquivo, _n_neighbor, _metric, fout, x, y, arquivo)
				fout.close

				stop = datetime.datetime.now()

			        ftime.write(str(_n_neighbor) + "-" + _metric + ". Start: " + str(start) + ". Stop: " + str(stop) +"." )
				ftime.write("\n")
		ftime.close
		print("Done")

        except KeyboardInterrupt:
	        rospy.loginfo("Shutting down")

if __name__ == "__main__":
        main(sys.argv)


