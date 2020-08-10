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
from sklearn.manifold import TSNE

def main(args):

	try:
	        rospy.init_node('n_lab_2_plot_feature', anonymous=True)
        	rospack = rospkg.RosPack()

		#Caminho do package
                ml_lab1_path = rospack.get_path("ml_lab2")
                ml_lab1_path += "/scripts"

		plt.grid(True)
		plt.xlabel("Features")
		plt.ylabel("Label")

		data_train = ml_lab1_path + "/data/train.txt"
		data_test =  ml_lab1_path + "/data/test.txt"

		plt.title("Train")
		X_train, y_train = load_svmlight_file(data_train)
		X_test, y_test = load_svmlight_file(data_test)

		model = TSNE(n_components=2, init='pca', random_state=0)
		transformed = model.fit_transform(X_test.todense())

		fig, ax = plt.subplots(figsize=(8,8))

		i = 0
		for g in numpy.unique(y_train):
			ix = numpy.where(y_train == g)
			ax.scatter(transformed[:,0][ix], transformed[:,1][ix], c=[plt.cm.tab10(float(g)/9)], s=9, label=str(i))
			i = i + 1

		plt.legend(loc='lower left',fontsize=7)
		plt.axhline(color='b')
		plt.axvline(color='b')
		#plt.savefig(train+".png")
		plt.show()

                print("Done")

        except KeyboardInterrupt:
	        rospy.loginfo("Shutting down")

if __name__ == "__main__":
        main(sys.argv)


