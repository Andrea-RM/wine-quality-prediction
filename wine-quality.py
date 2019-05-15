from tensorflow import keras
from sklearn.model_selection import cross_val_score, KFold
from tensorflow.keras.wrappers.scikit_learn import KerasRegressor
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.models import Sequential
from sklearn.metrics import accuracy_score
import pandas as pd
import numpy as np

feature_names = [
    "fixed acidity",
    "volatile acidity",
    "citric acid",
    "residual sugar",
    "chlorides",
    "free sulfur dioxide",
    "total sulfur dioxide",
    "density",
    "pH",
    "sulphates",
    "alcohol",
    "quality",
]

red_wine_data = pd.read_csv('winequality-red.csv',
                            names=feature_names, sep=";", header=1)
print(red_wine_data.head())
print(red_wine_data.describe())

red_wine_features = red_wine_data[feature_names].drop('quality', axis=1).values
red_wine_quality = red_wine_data['quality'].values


def create_model():
    model = Sequential()
    model.add(Dense(16, input_dim=11,
                    kernel_initializer='normal', activation='relu'))
    model.add(Dense(8, kernel_initializer='normal', activation='relu'))
    model.add(Dense(4, kernel_initializer='normal', activation='relu'))
    model.add(Dense(1, activation='linear'))
    model.compile(loss='mse', optimizer='adam', metrics=['accuracy'])
    return model


estimator = KerasRegressor(build_fn=create_model, nb_epoch=100, verbose=0)
estimator.fit(red_wine_features, red_wine_quality)
prediction = estimator.predict(red_wine_features)
train_error = np.abs(red_wine_quality - prediction)
print(np.mean(train_error))
