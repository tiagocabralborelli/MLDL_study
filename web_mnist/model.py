import tensorflow as tf
from tensorflow.keras.utils import to_categorical

def CNN():
    model = tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(filters = 128*3,kernel_size = 3, activation = "relu", input_shape = (28,28,1)),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Conv2D(filters = 64*3,kernel_size = 3, activation = "relu"),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Conv2D(filters = 32*3,kernel_size = 3, activation = "relu"),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dropout(0.42),
        tf.keras.layers.Dense(1000,activation = "relu"),
        tf.keras.layers.Dropout(0.42),            
        tf.keras.layers.Dense(500, activation = "relu"),
        tf.keras.layers.Dropout(0.42),
        tf.keras.layers.Dense(100, activation = "relu"),
        tf.keras.layers.Dropout(0.42),
        tf.keras.layers.Dense(10,   activation = "softmax"),
        ])
    model.compile(
        optimizer = tf.optimizers.Adam(),
        loss = tf.keras.losses.SparseCategoricalCrossentropy(),
        metrics = ["accuracy"]
    )
    return model

class Mnist:
    def __init__(self,xtrain,ytrain,xtest,ytest):
        self.XTrain = xtrain
        self.yTrain = ytrain
        self.XTest  = xtest
        self.yTest  = ytest

    def reshape(self):
        self.XTrain = self.XTrain.reshape(self.XTrain.shape[0],28,28,1).astype('float32') / 255.0
        self.XTest = self.XTest.reshape(self.XTest.shape[0],28,28,1).astype('float32') / 225.0
        return self.XTrain, self.XTest

    def encoding(self):
        self.yTrain = to_categorical(self.yTrain)
        self.yTest = to_categorical(self.yTest)
        return self.yTrain, self.yTest

