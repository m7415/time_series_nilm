# copie de https://machinelearningmastery.com/time-series-prediction-lstm-recurrent-neural-networks-python-keras/
print("imports...", end="",flush=True)
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' # pour désactiver les warnings de tensorflow

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import math
import tensorflow as tf
# from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from data_by_window import data_by_window

def print_window(x, y=None):
	for j in range(len(x)):
		print(f" {x[j]}", end = "")
		if y is not None:
			print(f" > {y[j]}", end = "")
		print()

win_size = 60
nb_data = 6959*60
taille_entrainement = 0.8
batch_size = 20



if nb_data != -1:
	print("done\nreading X_train.csv...", end="", flush=True)
	xtrain_csv = pd.read_csv("X_train.csv", usecols=["consumption"]).iloc[:nb_data]
	xtrain_csv = xtrain_csv.interpolate(method="linear")
	print("done\nreading y_train.csv...", end="", flush=True)
	ytrain_csv = pd.read_csv("y_train.csv", usecols=[1,2,3,4]).iloc[:nb_data]
	ytrain_csv = ytrain_csv.interpolate(method="linear")
	print("done")
else:
	print("done\nreading X_train.csv...", end="", flush=True)
	xtrain_csv = pd.read_csv("X_train.csv", usecols=["consumption"])
	xtrain_csv = xtrain_csv.interpolate(method="linear")
	print("done\nreading y_train.csv...", end="", flush=True)
	ytrain_csv = pd.read_csv("y_train.csv", usecols=[1,2,3,4])
	ytrain_csv = ytrain_csv.interpolate(method="linear")
	print("done")

concat = pd.concat( (xtrain_csv, ytrain_csv), axis=1 )
print(concat)

scaler = MinMaxScaler()
concat = scaler.fit_transform(concat)

print(concat)

xtrain_scaled = pd.DataFrame(concat[:,0], columns=["consumption"])
ytrain_scaled = pd.DataFrame(concat[:,1:5], columns=["washing_machine","fridge_freezer","TV","kettle"])


print("Building windows for x...", end="", flush=True)
x_win = data_by_window(xtrain_scaled, window_size=win_size, step = win_size)
print("done\nBuilding windows for y_fridge...", end="", flush=True)
# y_win = data_by_window(ytrain, window_size=10, features=["washing_machine","fridge_freezer","TV","kettle"], step = 10)
y_win_fridge = data_by_window(ytrain_scaled, window_size=win_size, features=["fridge_freezer"], step = 10)
print("done")

# for i in [0,1]:
# 	print(f"window {i} :")
# 	print_window(x_win[i], y_win_fridge[i])

# on fixe le random, pour pouvoir répéter nos résultats
tf.random.set_seed(0)


def create_dataset(x_win,y_win, taille_entrainement=0.8):
	t = int(0.8*len(x_win))
	
	return np.array(x_win[:t]),np.array(y_win[:t]), \
		   np.array(x_win[t:]), np.array(y_win[t:])

xtrain, ytrain, xtest, ytest = create_dataset(x_win,y_win_fridge, taille_entrainement)

# print(int(0.8*len(x_win)))
# print("last train window")
# print_window(xtrain[-1], ytrain[-1])
# print("first test window")
# print_window(xtest[0], ytest[0])

xtrain = np.reshape(xtrain, (xtrain.shape[0],1,xtrain.shape[1])) 
xtest = np.reshape(xtest, (xtest.shape[0],1,xtest.shape[1])) 

# notre réseau de neurones :
# win_size consommations en input
# une couche de 5 neurones LSTM
# win_size valeur en sortie (quelle part de consommation)
model_fridge = Sequential()
model_fridge.add(LSTM(200, input_shape=(1,win_size)))
# model_fridge.add(LSTM(100, return_series=True, input_shape=(1,win_size)))
# model_fridge.add(LSTM(100))
model_fridge.add(Dense(win_size)) # win_size output numérique
model_fridge.compile(loss="mean_squared_error", optimizer="adam")

# entrainement sur les données d'entrainement
print("Begin training")
model_fridge.fit(xtrain,
				 ytrain,
				 epochs=100,
				 batch_size=batch_size,
				 verbose=2)

model_fridge.summary()

# prédictions 
predic_train = model_fridge.predict(xtrain) # sur les données d'entrainement
predic_test  = model_fridge.predict(xtest)  # sur les données de test


# # calcul des erreurs
# erreur_train = np.sqrt( mean_squared_error(ytrain[:,:,0], predic_train) )
# erreur_test  = np.sqrt( mean_squared_error(ytest[:,:,0], predic_test) )

# print(f"erreur sur données d'entrainement : {erreur_train}")
# print(f"erreur sur données de test        : {erreur_test}")

absi_predic_train = []
ordo_predic_train = []
absi_predic_test = []
ordo_predic_test = []

absi_vrai = []
ordo_vrai = []

t = int(len(x_win)*taille_entrainement)


plt.figure("Résultats sur frigo")

# plt.plot(absi_vrai, ordo_vrai, "#0000FF")

plt.plot(ytrain_scaled["fridge_freezer"], "#0000FF")

plt.plot(predic_train.flat, "#00FF00")
plt.plot(range((t+1)*win_size, len(xtrain_scaled)),predic_test.flat, "#FF0000")

plt.show()
