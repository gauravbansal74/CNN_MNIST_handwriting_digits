# This is a sample Python script.

# Press ⌃R to execute it or replace it with your code.
# Press Double ⇧ to search everywhere for classes, files, tool windows, actions, and settings.

import BaseLineModel

def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press ⌘F8 to toggle the breakpoint.


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    x_train, y_train, x_test, y_test = BaseLineModel.load_dataset()
    print('Train: X=%s, y=%s' % (x_train.shape, y_train.shape))
    print('Test: X=%s, y=%s' % (x_test.shape, y_test.shape))
    BaseLineModel.plot_dataset(x_train, y_train, x_test, y_test)
    x_train, x_test = BaseLineModel.prepare_dataset(x_train, x_test)
    model = BaseLineModel.define_model()
    model.fit(x_train, y_train, epochs=10, batch_size=32, verbose=0)
    model.save('final_model.keras')


# See PyCharm help at https://www.jetbrains.com/help/pycharm/
