from model import Mnist, CNN
from keras.datasets import mnist
import numpy as np
(x_train, y_train), (x_test, y_test) = mnist.load_data()

clf = Mnist(
    xtrain = x_train, 
    ytrain = y_train,
    xtest = x_test,
    ytest = y_test)
xtrain, xtest = clf.reshape()
# ytrain, ytest = clf.encoding()
print(list(map(np.shape,[xtrain,xtest,y_train,y_test])))

model = CNN()
model.fit(
    xtrain, 
    y_train,
    epochs = 12,
    validation_data = (xtest, y_test),
    verbose = 1
)
model.save("mnist_model")
