import tensorflow as tf
from tensorflow.keras.layers import Dense,Conv1D,Flatten,ReLU
from tensorflow.keras import Model

class TDNN(Model):
    def __init__(self,k_size,units1,units2):
        super(TDNN,self).__init__()
        self.flat = Flatten()
        self.relu = ReLU()
        self.conv1 = Conv1D(units1,k_size)
        self.conv2 = Conv1D(units2,k_size,)
        self.dense = Dense(1,activation='sigmoid')
    def call(self,X):
        X = self.conv1(X)
        X = self.relu(X)
        X = self.conv2(X)
        X = self.relu(X)
        X = self.flat(X)
        X = self.dense(X)
        return X