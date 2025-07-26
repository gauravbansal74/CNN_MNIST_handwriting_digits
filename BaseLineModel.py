from tensorflow.keras.datasets import mnist
from tensorflow.keras.layers import Conv2D, Dense
from tensorflow.keras.optimizers.legacy import SGD
from tensorflow.keras.utils import to_categorical
from matplotlib import pyplot as plt
from tensorflow.python.keras import Sequential
from tensorflow.python.keras.layers import MaxPooling2D, Flatten


def load_dataset():
	# load dataset
	(x_train, y_train), (x_test, y_test) = mnist.load_data()
	# reshape dataset to have a single channel
	x_train = x_train.reshape((x_train.shape[0], 28, 28, 1))
	x_test = x_test.reshape((x_test.shape[0], 28, 28, 1))
	# one hot encode target values
	y_train = to_categorical(y_train)
	y_test = to_categorical(y_test)
	print(y_train)
	return x_train, y_train, x_test, y_test


def plot_dataset(x_train, y_train, x_test, y_test):
	plt.figure(figsize=(15, 5))
	for i in range(1,6):
		plt.subplot(1, 5, i)
		plt.imshow(x_train[i,:,:], cmap=plt.get_cmap('gray'))
		plt.axis('off')
	plt.show()


def prepare_dataset(train, test):
	# convert from integers to floats
	train_norm = train.astype('float32')
	test_norm = test.astype('float32')
	# normalize to range 0-1
	train_norm = train_norm / 255.0
	test_norm = test_norm / 255.0
	# return normalized images
	return train_norm, test_norm

# define cnn model
def define_model():
	model = Sequential()
	model.add(Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', input_shape=(28, 28, 1)))
	model.add(MaxPooling2D((2, 2)))
	model.add(Flatten())
	model.add(Dense(100, activation='relu', kernel_initializer='he_uniform'))
	model.add(Dense(10, activation='softmax'))
	# compile model
	opt = SGD(learning_rate=0.01, momentum=0.9)
	model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
	return model