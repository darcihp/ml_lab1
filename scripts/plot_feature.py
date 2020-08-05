#!/usr/bin/env python
# -*- encoding: iso-8859-1 -*-
import sys
import numpy
import rospy
import roslib
import pylab as pl
import rospkg
import matplotlib.pyplot as plt
from sklearn.datasets import load_svmlight_file
from sklearn.model_selection import train_test_split
import os
#from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import plot_confusion_matrix

def main(args):

	try:
	        rospy.init_node('n_lab_1_plot_feature', anonymous=True)
        	rospack = rospkg.RosPack()

		#Caminho do package
                ml_lab1_path = rospack.get_path("ml_lab1")
                ml_lab1_path += "/scripts"

                #Percorre diretório para ler arquivos
                for _, _, arquivos in os.walk(ml_lab1_path + "/features"): print("")

		#plt.xlim(0, 9000)
		#plt.ylim(0.8, 1)
		plt.grid(True)
		#plt.yticks(numpy.arange(0.8, 1.05, 0.05))
		plt.xlabel("Features")
		plt.ylabel("Label")

                for arquivo in arquivos:
			name = arquivo
			plt.title(name)
		        X_data, y_data = load_svmlight_file(ml_lab1_path + "/features/"+name)
			X_train, X_test, y_train, y_test =  train_test_split(X_data, y_data, test_size=0.5, random_state = 5)




			model = TSNE(n_components=2, init='pca', random_state=0)
			transformed = model.fit_transform(X_train.todense())

			fig, ax = plt.subplots(figsize=(8,8))

			i = 0
			for g in numpy.unique(y_train):
				ix = numpy.where(y_train == g)
				ax.scatter(transformed[:,0][ix], transformed[:,1][ix], c=[plt.cm.tab10(float(g)/9)], s=9, label=str(i))
				i = i + 1

			plt.legend(loc='lower left',fontsize=7)
			plt.axhline(color='b')
			plt.axvline(color='b')
			plt.savefig(name+".png")
			#plt.show()

                print("Done")

        except KeyboardInterrupt:
	        rospy.loginfo("Shutting down")

if __name__ == "__main__":
        main(sys.argv)


