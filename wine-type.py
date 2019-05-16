from tensorflow import keras
from sklearn.model_selection import cross_val_score, KFold
from keras.wrappers.scikit_learn import KerasClassifier
from keras.layers import Dense, Dropout
from keras.models import Sequential
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis

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

interested_features = [
    "fixed acidity",
    "volatile acidity",
    "residual sugar",
    "chlorides",
    "free sulfur dioxide",
    "total sulfur dioxide",
    "density",
    "sulphates",
]

red_wine_data = pd.read_csv('winequality-red.csv',
                            names=feature_names, sep=";", header=1)
white_wine_data = pd.read_csv(
    'winequality-white.csv', names=feature_names, sep=";", header=1)

red_wine_data['type'] = 0
white_wine_data['type'] = 1

wine_data = red_wine_data.append(white_wine_data)
wine_features = wine_data[interested_features].values
wine_type = wine_data['type'].values

scaler = StandardScaler().fit(wine_features)
wine_features = scaler.transform(wine_features)

corr = wine_data.corr()
ax = sns.heatmap(corr, vmin=-1, vmax=1, center=0,
                 cmap=sns.diverging_palette(20, 220, n=200),
                 square=True)
ax.set_xticklabels(
    ax.get_xticklabels(),
    rotation=45,
    horizontalalignment='right'
)
# plt.show()


def base_model():
    model = Sequential()
    model.add(Dense(512, input_dim=len(interested_features),
                    kernel_initializer='normal', activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(128, kernel_initializer='normal', activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(16, kernel_initializer='normal', activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy',
                  optimizer='adam', metrics=['accuracy'])
    return model


models = []
estimator = KerasClassifier(build_fn=base_model,
                            nb_epoch=100, verbose=0)
models.append(('NeuralNet', estimator))
models.append(('KNN', KNeighborsClassifier(3)))
models.append(('SVC', SVC(gamma=2, C=1)))
models.append(('GaussianProcess', GaussianProcessClassifier(1.0 * RBF(1.0))))
models.append(('DecisionTree', DecisionTreeClassifier(max_depth=5)))
models.append(('RandomForest', RandomForestClassifier(max_depth=5)))
models.append(('MLP', MLPClassifier(alpha=1, max_iter=1000)))
models.append(('AdaBoost', AdaBoostClassifier(3)))
models.append(('GaussianNB', GaussianNB(3)))
models.append(('QuadraticDiscrimination', QuadraticDiscriminantAnalysis()))


for name, model in models:
    results = cross_val_score(model, wine_features, wine_type, cv=10)
    print("{}: {}".format(name, results.mean()))
