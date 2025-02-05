import glob
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from tensorflow import keras
import os
import math
import firebase_admin
from firebase_admin import credentials, db, storage
from uuid import uuid4
from collections import Counter
from dotenv import load_dotenv

load_dotenv()

# Firebase 설정
cred = credentials.Certificate(os.getenv('CREDENTIALS_PATH'))
firebase_admin.initialize_app(cred, {
    'databaseURL': os.getenv('DATABASE_URL'),
    'storageBucket': os.getenv('STORAGE_BUCKET')
})

bucket = storage.bucket()

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

id_name = "test"  # ID 입력
file_path = 'D:/exhibition/' + id_name + ".csv"

# Firebase 업로드
def fileUpload(file):
    blob = bucket.blob(file)
    new_token = uuid4()
    metadata = {"firebaseStorageDownloadTokens": new_token}
    blob.metadata = metadata
    blob.upload_from_filename(filename=file, content_type='csv')
    return blob.public_url

# 감정별 컬러 리스트
color_lists = {
    1: ['#f9f3d3', '#dfedb0', '#a5bc6c', '#c9bca9', '#b9c2a5'],
    2: ['#8f8794', '#b299c3', '#cfaee7', '#c8b9be', '#988684'],
    3: ['#d7be2d', '#833765', '#5d3b6e', '#db9148', '#c06793'],
    4: ['#fdeded', '#eeced9', '#faeef2', '#f5eff9', '#cbb9b7'],
    5: ['#c2448d', '#f59701', '#f5a6a2', '#f3d900', '#ffe76b'],
    6: ['#cb003f', '#f3d900', '#01a46d', '#2c6fc0', '#007da3']
}

# 컬러 선택 
def color_pick(emotion_num, count):
    global color_list
    if emotion_num in color_lists:
        color_list.extend(color_lists[emotion_num][:count])
        color_lists[emotion_num] = color_lists[emotion_num][count:]

# 감정 매칭
def get_emotion(num):
    emotions = {1: "Natural", 2: "Elegant", 3: "Erotic",
                4: "Romantic", 5: "Casual", 6: "Dynamic"}
    return emotions.get(num, "Unknown")

# 데이터 로드 및 전처리
df = pd.read_csv(file_path).drop_duplicates(
    ['delta'], keep='first').iloc[1:, 1:9].astype(int)
signal_x = np.fft.fft(pdf1, axis=0) / len(df)
df_scaled = StandardScaler().fit_transform(pd.DataFrame(abs(signal_x).astype(float),
                                                        columns=['delta', 'theta', 'lowAlpha',
                                                                 'highAlpha', 'lowBeta', 'highBeta',
                                                                 'lowGamma', 'highGamma']))

# 모델 예측
new_model = keras.models.load_model('fft_model3.h5')
predictions = new_model.predict(df_scaled)

# 예측된 감정 값 저장
emotion_counts = Counter(np.argmax(predictions, axis=1) + 1) # axis=1 => predictions의 각 행에서 최댓값을 가진 인덱스
top_emotions = emotion_counts.most_common(6)
top3_emotions = top_emotions[:3]

# 감정별 데이터 정리
emotion_percentages = {
    emotion: value / df_scaled.shape[0] * 100 for emotion, value in top_emotions}

# 상위 3개 감정 비율 계산
percents = sum(value for _, value in top3_emotions) / 10
top3_percentages = {emotion: round(value / percents)
                    for emotion, value in top3_emotions}


total_top3 = sum(top3_percentages.values())

# 10으로 맞추기
if total_top3 > 10:
    top3_percentages[top3_emotions[2][0]] -= (total_top3 - 10)
elif total_top3 < 10:
    top3_percentages[top3_emotions[2][0]] += (10 - total_top3)

# 감정 최대 5로 제한
if top3_percentages[top3_emotions[0][0]] > 5:
    excess = top3_percentages[top3_emotions[0][0]] - 5
    top3_percentages[top3_emotions[0][0]] = 5
    top3_percentages[top3_emotions[1][0]] += excess

if top3_percentages[top3_emotions[1][0]] > 5:
    excess = top3_percentages[top3_emotions[1][0]] - 5
    top3_percentages[top3_emotions[1][0]] = 5
    top3_percentages[top3_emotions[2][0]] += excess

# 컬러 리스트 생성
color_list = []
for emotion, value in top3_percentages.items():
    half_count = math.ceil(value / 2)
    color_pick(emotion, half_count)
    color_pick(emotion, count - half_count)

emotion_all = [(get_emotion(emotion), round(percentage, 2))
               for emotion, percentage in emotion_percentages.items()]

emotion_top3_color = [(get_emotion(emotion), value)
                      for emotion, value in top3_percentages.items()],


fileUpload(file_path)

db.reference().update({id_name: {'emotion_all': emotion_all,
                                 'emotion_top3_color': emotion_top3_color, 'colorlist': color_list}})


print("\n감정 결과 (Top 3):", [get_emotion(emotion)
      for emotion, _ in top3_emotions])

print("\n상위 3가지 감정 유사율:")
for emotion, percentage in emotion_percentages.items():
    print(f"[ {get_emotion(emotion)} : {percentage:.2f}% ]")

print("\n첫 번째 감정, 컬러칩 개수:", get_emotion(
    top3_emotions[0][0]), top3_percentages[top3_emotions[0][0]])
print("두 번째 감정, 컬러칩 개수:", get_emotion(
    top3_emotions[1][0]), top3_percentages[top3_emotions[1][0]])
print("세 번째 감정, 컬러칩 개수:", get_emotion(
    top3_emotions[2][0]), top3_percentages[top3_emotions[2][0]])

print("\n컬러 리스트:", color_list)
