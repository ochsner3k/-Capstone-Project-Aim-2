import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical

def load_ecg5000(one_hot=False, test_size=0.2, random_state=42):
    from sklearn.datasets import fetch_openml
    data = fetch_openml("ECG5000", version=1, as_frame=True)
    df = data.frame

    X = df.iloc[:, :-1].values
    y = df.iloc[:, -1].astype(int).values

    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    if one_hot:
        y_train = to_categorical((y_train.flatten() - 1), num_classes=5)  
        y_test = to_categorical((y_test.flatten() - 1), num_classes=5)
        print("Applied one-hot encoding:", y_train.shape)


    return X_train, X_test, y_train, y_test



def load_heartbeat_dataset(one_hot=False, test_size=0.2, random_state=42):

    train_path = "data/mitbih_train.csv"
    test_path = "data/mitbih_test.csv"
    
    df_train = pd.read_csv(train_path, header=None)
    df_test = pd.read_csv(test_path, header=None)
    
    df = pd.concat([df_train, df_test])
    X = df.iloc[:, :-1].values
    y = df.iloc[:, -1].astype(int).values
    num_classes = 5

    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    if one_hot:
        y = to_categorical(y, num_classes=num_classes)

    X = X[..., np.newaxis] 

    return train_test_split(X, y, test_size=0.2, random_state=42)
