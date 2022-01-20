from sklearn.ensemble import RandomForestClassifier
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.ensemble import GradientBoostingClassifier
from xgboost import XGBClassifier
############################################
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from keras.models import Sequential
from keras.layers import LSTM, Dense, Activation
from keras.callbacks import EarlyStopping
import tensorflow as tf

from sklearn.model_selection import train_test_split
import keras
import pickle

from keras.layers import Dense, Conv2D, Flatten, Dropout
from keras import regularizers

class LSTM_Model():
    def __init__(self, time_steps, hidden_dim, n_epochs,
                 activation, loss,
                 data_dim=1, output_dim=3):
        self.model = Sequential()
        self.model_name = "LSTM"
        self.time_steps = time_steps
        self.hidden_dim = hidden_dim
        self.n_epochs = n_epochs
        self.activation = activation
        self.loss = loss
        self.data_dim = data_dim
        self.output_dim = output_dim
        return

    def build(self):
        # expected input batch shape: (batch_size, timesteps, data_dim)
        # the sample of index i in batch k is the follow-up for the sample i in batch k-1.
        self.model.add(
            LSTM(self.hidden_dim, activation=self.activation, return_sequences=True, input_shape=(self.time_steps, self.data_dim)))
        # self.model.add(LSTM(hidden_dim, activation=activation, return_sequences=True))
        self.model.add(LSTM(self.hidden_dim, activation=self.activation))
        self.model.add(Dense(self.output_dim, activation='softmax'))

        opt = tf.optimizers.Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
        self.model.compile(loss=self.loss, optimizer=opt, metrics=['accuracy'])
        return self.model

    def fit(self, x_train, y_train, batch_size, epochs, validation_data, callbacks):
        return self.model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, validation_data=validation_data,
                              callbacks=callbacks)

    def predict(self, x):
        if len(x.shape) == 2:
            x = x.reshape(x.shape[0], x.shape[1], 1)
        return self.model.predict(x)

    def reshape_dataset(self, x, y):
        if x is not None:
            if len(x.shape) == 2:
                x = x.reshape(x.shape[0], x.shape[1], 1)
        if len(y.shape) == 1:
            y = y.reshape(y.shape[0], 1)
        return x, y


class CNN_Model():
    def __init__(self, time_steps, hidden_dim, n_epochs,
                 activation, loss,
                 data_dim=1, output_dim=3):
        self.model = Sequential()
        self.model_name = "CNN"
        self.time_steps = time_steps
        self.hidden_dim = hidden_dim
        self.n_epochs = n_epochs
        self.activation = activation
        self.loss = loss
        self.data_dim = data_dim
        self.output_dim = output_dim
        return

    def build(self):
        # add model layers
        self.model.add(Conv2D(16, kernel_size=2, activation=self.activation, input_shape=(self.time_steps, self.data_dim, 1),
                              kernel_regularizer=regularizers.l2(0.001)))
        # self.model.add(Dropout(0.1))
        self.model.add(Conv2D(16, kernel_size=2, activation=self.activation, kernel_regularizer=regularizers.l2(0.001)))
        # self.model.add(Dropout(0.1))
        self.model.add(Conv2D(16, kernel_size=2, activation=self.activation, kernel_regularizer=regularizers.l2(0.001)))
        # self.model.add(Dropout(0.1))
        self.model.add(Flatten())
        self.model.add(Dense(self.output_dim, activation='softmax'))

        opt = tf.optimizers.Adam(lr=0.002, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
        self.model.compile(loss=self.loss, optimizer=opt, metrics=['accuracy'])
        return self.model

    def fit(self, x_train, y_train, batch_size, epochs, validation_data, callbacks):
        return self.model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, validation_data=validation_data,
                              callbacks=callbacks)

    def predict(self, x):
        if x is not None:
            if len(x.shape) == 3:
                x = x.reshape(x.shape[0], x.shape[1], x.shape[2], 1)
        return self.model.predict(x)

    def reshape_dataset(self, x, y):
        if x is not None:
            if len(x.shape) == 3:
                x = x.reshape(x.shape[0], x.shape[1], x.shape[2], 1)
        if len(y.shape) == 1:
            y = y.reshape(y.shape[0], 1)
        return x, y

class RandomForest():
    def __init__(self,max_depth=36, n_estimators=100, max_features=0.2, criterion = 'gini'):
        self.model = RandomForestClassifier(n_estimators=n_estimators, criterion = criterion, max_depth=max_depth,max_features=max_features)
        self.model_name = "RandomForest"
    def fit(self,X,Y):
        return self.model.fit(X,Y)
    def predict(self,x):
        if len(x.shape) == 3:
            x = x.reshape(x.shape[0], x.shape[1] * x.shape[2])
        return self.model.predict(x)
    def reshape_dataset(self, X, Y):
        if X is not None:
            if len(X.shape) == 3:
                X = X.reshape(X.shape[0], X.shape[1]*X.shape[2])
        if len(Y.shape) == 1:
            Y = Y.reshape(Y.shape[0],1)
        return X, Y

class GradientBoost():
    def __init__(self, n_estimators=500,max_depth=50,learning_rate=0.05):
        self.model = GradientBoostingClassifier(n_estimators=n_estimators,max_depth=max_depth,learning_rate=learning_rate)
        self.model_name = "GradientBoost"
    def fit(self,X,Y):
        return self.model.fit(X,Y)
    def predict(self,x):
        if len(x.shape) == 3:
            x = x.reshape(x.shape[0], x.shape[1] * x.shape[2])
        return self.model.predict(x)
    def reshape_dataset(self, x, y):
        if x is not None:
            if len(x.shape) == 3:
                x = x.reshape(x.shape[0], x.shape[1]*x.shape[2])
        if len(y.shape) != 1:
            y = np.where(y==1)[1]
            y = y.reshape(y.shape[0],1)
        return x, y

class XGBoost():
    def __init__(self, n_estimators=100, max_depth=50,learning_rate=0.1,reg_lambda=0.1, verbose=False):
        self.model =XGBClassifier(n_estimators=n_estimators, max_depth=max_depth,learning_rate=learning_rate,reg_lambda=reg_lambda, verbose=verbose)
        self.model_name = "XGBoost"
    def fit(self,X,Y):
        return self.model.fit(X,Y)
    def predict(self,x):
        if len(x.shape) == 3:
            x = x.reshape(x.shape[0], x.shape[1] * x.shape[2])
        return self.model.predict(x)
    def reshape_dataset(self, x, y):
        if x is not None:
            if len(x.shape) == 3:
                x = x.reshape(x.shape[0], x.shape[1]*x.shape[2])
        if len(y.shape) != 1:
            y = np.where(y==1)[1]
            y = y.reshape(y.shape[0],1)
        return x, y
