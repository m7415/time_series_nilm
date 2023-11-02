import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

PATH_XTRAIN = "X_train.csv"
PATH_YTRAIN = "y_train.csv"
PATH_XTEST =  "X_test.csv"



def ajout_colonne(xtrain):
    # 7 booleens pour les 7 jours de la semaine
    for i,jour in enumerate(["mon","tue","wed","thu","fri","sat","sun"]):
        xtrain[jour] = xtrain["time_step"].apply(lambda x: x.dayofweek == i)
    xtrain["weekend"] = xtrain["time_step"].apply(lambda x: x.dayofweek >=5 )

    deg_per_minute = 0.25

    xtrain["circ_hour"] = xtrain["time_step"].apply(lambda x: x.hour*15 + x.minute*deg_per_minute)
    xtrain["cos_hour"] = xtrain["circ_hour"].apply(lambda x: np.cos(np.deg2rad(x)))
    xtrain["sin_hour"] = xtrain["circ_hour"].apply(lambda x: np.sin(np.deg2rad(x)))


# tests
if __name__ == "__main__":
    xtrain = pd.read_csv(PATH_XTRAIN, parse_dates=["time_step"])
    ytrain = pd.read_csv(PATH_YTRAIN, parse_dates=["time_step"])
    n = len(xtrain)

    ajout_colonne(xtrain)

    print("verif jours et weekend:")
    for i in range(0,60*24*10, 60*24):
        print(f"{i:5}: {xtrain['time_step'][i]} > ", end ="")
        for j in ["mon","tue","wed","thu","fri","sat","sun"]:
            if xtrain[j][i]:
                print(j, end="")
        print(f"  week-end ? {'oui' if xtrain['weekend'][i] else 'non'}")
    
    print()
    print("verif heure :")
    for i in range(50,70):
        print(f"{i}: {xtrain['time_step'][i]}")
        print(f"  {xtrain['circ_hour'][i]} ")
        print(f"  cos = {xtrain['cos_hour'][i]:.2}")
        print(f"  sin = {xtrain['sin_hour'][i]:.2}")