# make a prediction for a new image.
from numpy import argmax
from tensorflow.keras.utils import load_img
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
import numpy as np

# load and prepare the image
def load_image(filename):
        # load the image
        img = load_img(filename, target_size=(28, 28))
        # convert to array
        img = img_to_array(img)
        # convert to grayscale
        img = np.dot(img[...,:3], [0.2989, 0.5870, 0.1140])
        # reshape into a single sample with 1 channel
        img = img.reshape(1, 28, 28, 1)
        # prepare pixel data
        img = img.astype('float32')
        img = img / 255.0
        return img

# load an image and predict the class
def run_example():
        # load the image
        img = load_image('sample_image.png')
        print(img.shape)
        # load model
        model = load_model('final_model.keras')
        # predict the class
        predict_value = model.predict(img)
        print(predict_value)
        digit = argmax(predict_value)
        print(digit)

# entry point, run the example
run_example()