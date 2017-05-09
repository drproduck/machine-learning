import pandas as pd
import numpy as np

def process(drop):
    df = pd.read_csv("/home/drproduck/Downloads/creditcard.csv")
    if drop:
        num_features = 10
        features = ['V7', 'V10', 'V18', 'V4', 'V9', 'V16', 'V14', 'V11', 'V17', 'V12', 'Class']
        df = df[features]

    else: num_features = 30

    df = df.as_matrix()
    print(np.shape(df))

    #df[:, :num_features] = (df[:,:num_features] - df[:,:num_features].mean(axis=0))/df[:,num_features].var(axis=0)
    fraud = df[np.where(df[:,num_features] == 1)[0],:]
    normal = df[np.where(df[:,num_features] == 0)[0],:]

    np.save('fraud', fraud)
    np.save('normal', normal)
    np.savetxt('fraud', fraud)

process(True)