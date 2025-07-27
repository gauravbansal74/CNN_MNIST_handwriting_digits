# This is a sample Python script.
import numpy as np
# Press ⌃R to execute it or replace it with your code.
# Press Double ⇧ to search everywhere for classes, files, tool windows, actions, and settings.
from tensorflow.keras.models import load_model
import cv2
from tensorflow.python.framework.test_ops import in_polymorphic_twice

import BaseLineModel
import input_data

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    input_data.capture()
    # x_train, y_train, x_test, y_test = BaseLineModel.load_dataset()
    # print('Train: X=%s, y=%s' % (x_train.shape, y_train.shape))
    # print('Test: X=%s, y=%s' % (x_test.shape, y_test.shape))
    # BaseLineModel.plot_dataset(x_train, y_train, x_test, y_test)
    # x_train, x_test = BaseLineModel.prepare_dataset(x_train, x_test)
    # model = BaseLineModel.define_model()
    # print('before model fit')
    # model.fit(x_train, y_train, epochs=10, batch_size=32)
    # model.save('final_model.keras')
    # model = load_model('final_model.keras')
    # # evaluate model on test dataset
    # _, acc = model.evaluate(x_test, y_test, verbose=0)
    # print('> %.3f' % (acc * 100.0))
