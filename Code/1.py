# Plotting few simple images from the MNIST dataset
from keras.dataset import mnist
import matplotlib.pyplot as plt
# load /downloading the MNIST dataset
(X_train, y_train), (X_test, y_test) = mnist.load_data()
# plot 4 of images as gray scale
plt.subplot(221)
plt.imshow(X_train[0], cmap=plt.get_cmap('gray'))
plt.subplot(222)
plt.imshow(X_train[1], cmap=plt.get_cmap('gray'))
plt.subplot(223)
plt.imshow(X_train[2], cmap=plt.get_cmap('gray'))
plt.subplot(224)
plt.imshow(X_train[3], cmap=plt.get_cmap('gray'))
# show the plot
plt.show()