# -*- encoding: iso-8859-1 -*-
import sys
import numpy
import rospy
import roslib
import pylab as pl
import rospkg
import matplotlib.pyplot as plt

def main(args):

	try:
	        rospy.init_node('n_lab_1_knn', anonymous=True)
        	rospack = rospkg.RosPack()

		#Arquivos para gerar plot
		_euclideans = ["results_1_euclidean", "results_3_euclidean",  "results_5_euclidean",  "results_7_euclidean",  "results_9_euclidean",  "results_11_euclidean",  "results_15_euclidean"]
		_manhattans = ["results_1_manhattan", "results_3_manhattan",  "results_5_manhattan",  "results_7_manhattan",  "results_9_manhattan",  "results_11_manhattan",  "results_15_manhattan"]
		_chebyshevs = ["results_1_chebyshev", "results_3_chebyshev",  "results_5_chebyshev",  "results_7_chebyshev",  "results_9_chebyshev",  "results_11_chebyshev",  "results_15_chebyshev"]
		_minkowskis = ["results_1_minkowski", "results_3_minkowski",  "results_5_minkowski",  "results_7_minkowski",  "results_9_minkowski",  "results_11_minkowski",  "results_15_minkowski"]

		_metrics = [_euclideans, _manhattans, _chebyshevs, _minkowskis]

		_color = numpy.array(["red", "green", "blue", "pink", "magenta", "cyan", "orange"])

		#Caminho do package
		ml_lab1_path = rospack.get_path("ml_lab1")
		ml_lab1_path += "/scripts"

		for _metric in _metrics:

			plt.xlim(0, 2500)
			plt.ylim(0.8, 1)
			plt.grid(True)
			plt.yticks(numpy.arange(0.8, 1.05, 0.05))
			plt.xlabel("Number of characteristics")
			plt.ylabel("Accuracy[%]")

			i = 0
			for _n_neighbors in _metric:
				if i == 0:
					global name
					name = _n_neighbors.split("_")[2]
					plt.title(name)
					if name == "chebyshev":
						plt.ylim(0.0, 0.4)
						plt.yticks(numpy.arange(0.0, 0.4, 0.05))

				fout = open(ml_lab1_path +"/" + _n_neighbors, "r")
				x = []
				y = []
				accuracy = []
				features = []

				for line in fout:
					_x = float(line.split()[0])
					x.append(_x)
					_y = float(line.split()[1])
					y.append(_y)
					_f = _x * _y
					features.append(_f)
					accuracy.append(float(line.split()[2]))
				fout.close

				plt.scatter(features, accuracy, marker='o', c=_color[i], label="K="+_n_neighbors.split("_")[1])
				i = i + 1
			plt.legend(loc="upper left", frameon=True, ncol=4)
			plt.tight_layout()
			plt.savefig(name+"2.png")
			plt.show()

        except KeyboardInterrupt:
	        rospy.loginfo("Shutting down")

if __name__ == "__main__":
        main(sys.argv)


