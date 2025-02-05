import glob
import numpy as np
import pandas as pd
import time
import os
from sklearn.preprocessing import MinMaxScaler, StandardScaler

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

scaler = MinMaxScaler()
start = time.process_time()


# 감정별 데이터 경로
emotion_paths = {
    1: "D:/eeg_classification/natural/*.csv",
    2: "D:/eeg_classification/elegant/*.csv",
    3: "D:/eeg_classification/erotic/*.csv",
    4: "D:/eeg_classification/romantic/*.csv",
    5: "D:/eeg_classification/casual/*.csv",
    6: "D:/eeg_classification/dynamic/*.csv",
}

# 데이터 처리


def process_data(filename, label):
    df = pd.read_csv(filename).drop_duplicates(
        ['delta'], keep='first').iloc[1:, 1:9].astype(float)
    signal_x = np.fft.fft(df, axis=0) / len(df)
    df_fft = pd.DataFrame(abs(signal_x), columns=[
        'delta', 'theta', 'lowAlpha', 'highAlpha', 'lowBeta', 'highBeta', 'lowGamma', 'highGamma'])
    df_fft['label'] = label
    return df_fft


dataset_list = []
for label, path in emotion_paths.items():
    files = glob.glob(path)
    for filename in files:
        dataset_list.append(process_data(filename, label))

dataset = pd.concat(dataset_list)
dataset.to_csv('datasetfft.csv')
