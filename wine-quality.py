from tensorflow import keras
from sklearn.model_selection import cross_val_score, KFold
from keras.wrappers.scikit_learn import KerasRegressor
from keras.layers import Dense, Dropout
from keras.models import Sequential
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.svm import SVR

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
white_wine_data = pd.read_csv(
    'winequality-white.csv', names=feature_names, sep=";", header=1)

wine_data = red_wine_data.append(white_wine_data)
wine_features = wine_data[feature_names].drop('quality', axis=1).values
wine_quality = wine_data['quality'].values

scaler = StandardScaler().fit(wine_features)
wine_features_scaled = scaler.transform(wine_features)


def base_model():
    model = Sequential()
    model.add(Dense(1024, input_dim=11,
                    kernel_initializer='normal', activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(256, kernel_initializer='normal', activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(128, kernel_initializer='normal', activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(64, kernel_initializer='normal', activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(16, kernel_initializer='normal', activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(1))
    model.compile(loss='mse', optimizer='adam', metrics=['mae'])
    return model


models = []
estimator = KerasRegressor(build_fn=base_model,
                           nb_epoch=50, verbose=0)
models.append(('NeuralNet', estimator))
models.append(('DecisionTree', DecisionTreeRegressor()))
models.append(('RandomForest', RandomForestRegressor(n_estimators=100)))
models.append(('GradienBoost', GradientBoostingRegressor()))
models.append(('SVR', SVR(gamma='auto')))

for name, model in models:
    kfold = KFold(n_splits=5, random_state=43)
    results = np.sqrt(-1 * cross_val_score(model, wine_features_scaled,
                                           wine_quality, scoring='neg_mean_squared_error', cv=kfold))
    print("{}: {}".format(name, results.mean()))
