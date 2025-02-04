import glob
import numpy as np
import pandas as pd
import time
import os
from sklearn.preprocessing import MinMaxScaler, StandardScaler

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

scaler = MinMaxScaler()
start = time.process_time()


natural_files = glob.glob("D:\eeg_classification/natural/*.csv")
elegant_files = glob.glob("D:\eeg_classification/elegant/*.csv")
erotic_files = glob.glob("D:\eeg_classification/erotic/*.csv")
romantic_files = glob.glob("D:\eeg_classification/romantic/*.csv")
casual_files = glob.glob("D:\eeg_classification/casual/*.csv")
dynamic_files = glob.glob("D:\eeg_classification/dynamic/*.csv")


dataset = pd.DataFrame(columns=['delta','theta', 'lowAlpha', 'highAlpha','lowBeta', 'highBeta', 'lowGamma', 'highGamma','label']);


for filename in natural_files:
    df = pd.read_csv(filename)

    df1 = df.drop_duplicates(['delta'], keep='first')
    df1 = df1.drop(index=0)
    df1 = df1.iloc[:, 1:9].astype(int)  # numarr


    signal_x = np.fft.fft(df1 ,axis=0) / len(df1)
    df1_d = pd.DataFrame(abs(signal_x).astype(float), columns=['delta','theta', 'lowAlpha', 'highAlpha','lowBeta', 'highBeta', 'lowGamma', 'highGamma'])


    df1_d['label'] = 1

    dataset = pd.concat([dataset, df1_d])


for filename in elegant_files:
    df = pd.read_csv(filename)

    df1 = df.drop_duplicates(['delta'], keep='first')
    df1 = df1.drop(index=0)
    df1 = df1.iloc[:, 1:9].astype(int)  # numarr

    signal_x = np.fft.fft(df1, axis=0) / len(df1)
    df1_d = pd.DataFrame(abs(signal_x).astype(float), columns=['delta','theta', 'lowAlpha', 'highAlpha','lowBeta', 'highBeta', 'lowGamma', 'highGamma'])

    df1_d['label'] = 2

    dataset = pd.concat([dataset, df1_d])

for filename in erotic_files:
    df = pd.read_csv(filename)

    df1 = df.drop_duplicates(['delta'], keep='first')
    df1 = df1.drop(index=0)
    df1 = df1.iloc[:, 1:9].astype(int)  # numarr

    signal_x = np.fft.fft(df1, axis=0) / len(df1)
    df1_d = pd.DataFrame(abs(signal_x).astype(float), columns=['delta','theta', 'lowAlpha', 'highAlpha','lowBeta', 'highBeta', 'lowGamma', 'highGamma'])

    df1_d['label'] = 3

    dataset = pd.concat([dataset, df1_d])

for filename in romantic_files:
    df = pd.read_csv(filename)

    df1 = df.drop_duplicates(['delta'], keep='first')
    df1 = df1.drop(index=0)
    df1 = df1.iloc[:, 1:9].astype(int)  # numarr

    signal_x = np.fft.fft(df1, axis=0) / len(df1)
    df1_d = pd.DataFrame(abs(signal_x).astype(float), columns=['delta','theta', 'lowAlpha', 'highAlpha','lowBeta', 'highBeta', 'lowGamma', 'highGamma'])

    df1_d['label'] = 4

    dataset = pd.concat([dataset, df1_d])

for filename in casual_files:
    df = pd.read_csv(filename)

    df1 = df.drop_duplicates(['delta'], keep='first')
    df1 = df1.drop(index=0)
    df1 = df1.iloc[:, 1:9].astype(int)  # numarr

    signal_x = np.fft.fft(df1, axis=0) / len(df1)
    df1_d = pd.DataFrame(abs(signal_x).astype(float), columns=['delta','theta', 'lowAlpha', 'highAlpha','lowBeta', 'highBeta', 'lowGamma', 'highGamma'])

    df1_d['label'] = 5

    dataset = pd.concat([dataset, df1_d])

for filename in dynamic_files:
    df = pd.read_csv(filename)

    df1 = df.drop_duplicates(['delta'], keep='first')
    df1 = df1.drop(index=0)
    df1 = df1.iloc[:, 1:9].astype(int)  # numarr

    signal_x = np.fft.fft(df1, axis=0) / len(df1)
    df1_d = pd.DataFrame(abs(signal_x).astype(float), columns=['delta','theta', 'lowAlpha', 'highAlpha','lowBeta', 'highBeta', 'lowGamma', 'highGamma'])

    df1_d['label'] = 6

    dataset = pd.concat([dataset, df1_d])

dataset.to_csv('datasetfft.csv')