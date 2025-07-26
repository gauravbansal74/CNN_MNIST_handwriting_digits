# MNISTDataSet.py

from tensorflow.keras.datasets import mnist
from matplotlib import pyplot as plt

#load dataset
def load_data():
    (trainX, trainY), (testX, testY) = mnist.load_data()
    print('Train: X=%s, y=%s' % (trainX.shape, trainY.shape))
    print('Test: X=%s, y=%s' % (testX.shape, testY.shape))
    for i in range(9):
            plt.subplot(3, 3, i+1)
            plt.imshow(trainX[i], cmap=plt.get_cmap('gray'))
    plt.show()
