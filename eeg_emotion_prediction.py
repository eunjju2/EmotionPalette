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


cred = credentials.Certificate('emotion-palette-firebase-adminsdk-ns3sc-c1ffbdddc6.json')
firebase_admin.initialize_app(cred, {
    'databaseURL': 'https://emotion-palette-default-rtdb.firebaseio.com/',
    'storageBucket': 'emotion-palette.appspot.com'})

bucket = storage.bucket()

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

id_name = "test" #ID 입력
filereadname = 'D:/exhibition/' + id_name + ".csv"
filename = id_name + ".csv"


def fileUpload(file):
    blob = bucket.blob(file)
    new_token = uuid4()
    metadata = {"firebaseStorageDownloadTokens": new_token}
    blob.metadata = metadata
    blob.upload_from_filename(filename=file, content_type='csv')
    print(blob.public_url)

arr = []

new_model = keras.models.load_model('fft_model3.h5')

color_list1 = ['#f9f3d3', '#dfedb0', '#a5bc6c', '#c9bca9', '#b9c2a5']
color_list2 = ['#8f8794', '#b299c3', '#cfaee7', '#c8b9be', '#988684']
color_list3 = ['#d7be2d', '#833765', '#5d3b6e', '#db9148', '#c06793']
color_list4 = ['#fdeded', '#eeced9', '#faeef2', '#f5eff9', '#cbb9b7']
color_list5 = ['#c2448d', '#f59701', '#f5a6a2', '#f3d900', '#ffe76b']
color_list6 = ['#cb003f', '#f3d900', '#01a46d', '#2c6fc0', '#007da3']

color_list = []


def color_pick1(n):
    for i in range(n):
        color_list.append(color_list1[i])
    del color_list1[0:n]


def color_pick2(n):
    for i in range(n):
        color_list.append(color_list2[i])
    del color_list2[0:n]


def color_pick3(n):
    for i in range(n):
        color_list.append(color_list3[i])
    del color_list3[0:n]


def color_pick4(n):
    for i in range(n):
        color_list.append(color_list4[i])
    del color_list4[0:n]


def color_pick5(n):
    for i in range(n):
        color_list.append(color_list5[i])
    del color_list5[0:n]


def color_pick6(n):
    for i in range(n):
        color_list.append(color_list6[i])
    del color_list6[0:n]


def aaa(num, c):
    if num == 1:
        color_pick1(c)
    elif num == 2:
        color_pick2(c)
    elif num == 3:
        color_pick3(c)
    elif num == 4:
        color_pick4(c)
    elif num == 5:
        color_pick5(c)
    elif num == 6:
        color_pick6(c)


def emotion(num):
    if num == 1:
        return "Natural"
    elif num == 2:
        return "Elegant"
    elif num == 3:
        return "Erotic"
    elif num == 4:
        return "Romantic"
    elif num == 5:
        return "Casual"
    elif num == 6:
        return "Dynamic"


pre_df = pd.read_csv(filereadname)
pdf1 = pre_df.drop_duplicates(['delta'], keep='first')
pdf1 = pdf1.drop(index=0)
pdf1 = pdf1.iloc[:, 1:9].astype(int)  # numarr

signal_x = np.fft.fft(pdf1, axis=0) / len(pdf1)
df1_d = pd.DataFrame(abs(signal_x).astype(float),
                     columns=['delta', 'theta', 'lowAlpha',
                              'highAlpha', 'lowBeta', 'highBeta',
                              'lowGamma', 'highGamma'])

df_sc = StandardScaler().fit_transform(df1_d)
print(df_sc.shape)
predictions = new_model.predict(df_sc)
for i in range(df_sc.shape[0]):
    arr.append(np.argmax(predictions[i]) + 1)

count_items = Counter(arr)
count_items_top = count_items.most_common(n=6)


print("-------------------")
print(arr)
print(count_items)
print("-------------------")

first = count_items_top[0][0]  # 감정
second = count_items_top[1][0]
third = count_items_top[2][0]
fourth = count_items_top[3][0]
fifth = count_items_top[4][0]
sixth = count_items_top[5][0]

first_var = count_items_top[0][1]  # 값
second_var = count_items_top[1][1]
third_var = count_items_top[2][1]
fourth_var = count_items_top[3][1]
fifth_var = count_items_top[4][1]
sixth_var = count_items_top[5][1]

first_rate = first_var / df_sc.shape[0] * 100  # 유사율
second_rate = second_var / df_sc.shape[0] * 100
third_rate = third_var / df_sc.shape[0] * 100
fourth_rate = fourth_var / df_sc.shape[0] * 100
fifth_rate = fifth_var / df_sc.shape[0] * 100
sixth_rate = sixth_var / df_sc.shape[0] * 100

percent = (first_var + second_var + third_var) / 10
percent2 = (first_var + second_var + third_var) / 100

first_per = round(first_var / percent)
second_per = round(second_var / percent)
third_per = round(third_var / percent)


all = first_per + second_per + third_per
if all > 10:
    third_per = third_per - (all-10)

if first_per >= 5:
    first_per = 5
    second_per += first_per - 5
    if second_per > 5:
        third_per += second_per - 5

if all < 10 :
    third_per = third_per + (10-all)


first_cnt = math.ceil(first_per / 2)
second_cnt = math.ceil(second_per / 2)
third_cnt = math.floor(third_per / 2)

aaa(first, first_cnt)
aaa(second, second_cnt)
aaa(third, third_cnt)
aaa(first, first_per - first_cnt)
aaa(second, second_per - second_cnt)
aaa(third, third_per - third_cnt)

emotion_all = [(emotion(first), (first_var / first_var) * 100),
               (emotion(second), (second_var / first_var) * 100),
               (emotion(third), (third_var / first_var) * 100),
               (emotion(fourth), (fourth_var / first_var) * 100),
               (emotion(fifth), (fifth_var / first_var) * 100),
               (emotion(sixth), (sixth_var / first_var) * 100)]

emotion_top3 = [emotion(first), emotion(second), emotion(third)]

emotion_top3_rate = [[emotion(first), (first_var / percent2)],
                     [emotion(second), (second_var / percent2)],
                     [emotion(third), (third_var / percent2)]]

emotion_top3_color = [(emotion(first), first_per),
                      (emotion(second), second_per),
                      (emotion(third), third_per)]

                                                                             emotion(fourth),round((fourth_var / first_var)*10),emotion(fifth),round((fifth_var / first_var)*10),emotion(sixth),round((sixth_var / first_var)*10)))
print("[ %s : %f ]\n[ %s : %f ]\n[ %s : %f ]\n[ %s : %f ]\n[ %s : %f ]\n[ %s : %f ] \n" % (
emotion(first), (first_var / first_var) * 100, emotion(second), (second_var / first_var) * 100, emotion(third),
(third_var / first_var) * 100,
emotion(fourth), (fourth_var / first_var) * 100, emotion(fifth), (fifth_var / first_var) * 100, emotion(sixth),
(sixth_var / first_var) * 100))

print("감정 결과 : '%s' , '%s' , '%s' \n" % (emotion(first), emotion(second), emotion(third)))

print('상위 3가지 감정 유사율 :\n[ %s : %f ]\n[ %s : %f ]\n[ %s : %f ] \n' % (
emotion(first), (first_var / percent2), emotion(second), (second_var / percent2), emotion(third),
(third_var / percent2)))
print('첫번째 감정 , 컬러칩 개수 : %s , %d \n두번째 감정 , 컬러칩 개수 : %s , %d \n세번째 감정 , 컬러칩 개수 : %s , %d \n ' % (
emotion(first), first_per, emotion(second), second_per, emotion(third), third_per))


print(color_list)
fileUpload(filereadname)

ref = db.reference()
ref.update({id_name: {'emotion_all': emotion_all, 'emotion_top3_color': emotion_top3_color, 'colorlist': color_list}})
