import pandas as pd
import numpy as np

def process(drop):
    df = pd.read_csv("/home/drproduck/Downloads/creditcard.csv")
    if drop:
        num_features = 19
        df = df.drop(['V28', 'V27', 'V26', 'V25', 'V24', 'V23', 'V22', 'V20', 'V15', 'V13', 'V8'], axis=1)

    else: num_features = 30

    df = df.as_matrix()
    print(np.shape(df))

    df[:, :num_features] = (df[:,:num_features] - df[:,:num_features].mean(axis=0))/df[:,num_features].var(axis=0)
    fraud = df[np.where(df[:,num_features] == 1)[0],:]
    normal = df[np.where(df[:,num_features] == 0)[0],:]

    np.save('fraud', fraud)
    np.save('normal', normal)
    np.savetxt('fraud', fraud)

process(False)