#!/usr/bin/env python
import numpy as np
import roslib
import sys
import cv2
import rospy
import os
import random
import rospkg

image_names = []
labels = []
cv_images = []
cv_resize = []

min_x = sys.maxint
min_y = sys.maxint
max_x = 0
max_y = 0

class Feature():
	def __init__(self, X, Y):
        	self._X = X
	        self._Y = Y

def load_vectors(path_digits):
	print('Loading labels...')
	arq = open(path_digits + '/files.txt')
	lines = arq.readlines()

	for line in lines:
		aux = line.split('/')[1]
		#Cria vetor com nome das imagens
		image_name = aux.split(' ')[0]
		image_names.append(image_name)

		#Cria vetor com label das imagens
		label = line.split(' ')[1]
		label = label.split('\n')[0]
		labels.append(label)

		#Criar vetor com imagens
		image = cv2.imread(path_digits +'/data/'+ image_name, 0)
		y, x = image.shape

		global max_x, max_y, min_x, min_y

		if x > max_x:
			max_x = x
		if y > max_y:
			max_y = y
		if x < min_x:
			min_x = x
		if y < min_y:
			min_y = y

		cv_images.append(image)

	max_x = 100
	max_y = 100
	min_x = 1
	min_y = 1

	for x in range(min_x, max_x, 5):
		for y in range(min_y, max_y, 5):
			cv_resize.append(Feature(x, y))

	#print(len(cv_resize))
	print("Vectors loaded...")
	print("max_x: %d, max_y: %d, min_x: %d, min_y: %d" %(max_x, max_y, min_x, min_y))


#########################################################
# Usa o valor dos pixels como caracteristica
#
#########################################################

def rawpixel(image, label, fout, X, Y):

	## novas dimensoes
	#X= 200
	#Y= 100

	image = cv2.resize(image, (X,Y) )
	#cv2.imshow("image", image )
	#cv2.waitKey(0)
	#cv2.destroyAllWindows()

	fout.write(str(label) +  " ")

	indice = 0
	for i in range(Y):
		#vet= []
		for j in range(X):
			if( image[i][j] > 128):
				v = 0
			else:
				v = 1
			#vet.append(v)

			fout.write(str(indice)+":"+str(v)+" ")
			indice = indice+1

	fout.write("\n")


def main(args):

	rospy.init_node('n_lab_1', anonymous=True)
	rospack = rospkg.RosPack()

	try:
		#Caminho do package
		ml_lab1_path = rospack.get_path("ml_lab1")
		ml_lab1_path += "/scripts"

		#Carrega vetores
		load_vectors(ml_lab1_path +'/digits')

		print("Vector Size: %d" %(len(cv_resize)))
		#Cria Features
		for i in range(len(cv_resize)):
			print("Inter: %d" %(i))
			fout = open(ml_lab1_path + "/features/f_" + str(i) + "-" + str(cv_resize[i]._X)+"-"+ str(cv_resize[i]._Y) ,"w")
			for cv_image in range(len(cv_images)):
				rawpixel(cv_images[cv_image], labels[cv_image], fout, cv_resize[i]._X, cv_resize[i]._Y)
			fout.close

	except KeyboardInterrupt:
		rospy.loginfo("Shutting down")

if __name__ == "__main__":
	main(sys.argv)

