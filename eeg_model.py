import numpy as np 
import pandas as pd  
import matplotlib.pyplot as plt
import os
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
from tensorflow import keras
from tensorflow.keras import callbacks, layers
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import plot_model

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        tf.config.experimental.set_memory_growth(gpus[0], True)
    except RuntimeError as e:
        print(e)


for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


df = pd.read_csv('dataset0913.csv')

y = pd.get_dummies(df['label'])

top_fea = ['delta', 'theta', 'lowAlpha', 'highAlpha','lowBeta', 'highBeta', 'lowGamma', 'highGamma']

df_sc = StandardScaler().fit_transform(df[top_fea])

Xtr, xte, Ytr, yte = train_test_split(df_sc, y,  test_size=0.1)
xtr, xval, ytr, yval = train_test_split(Xtr, Ytr,  test_size=0.1)

model = keras.Sequential([
    layers.Dense(128, input_dim=(len(df_sc[0])), activation='relu'),
    layers.Dense(256, activation='relu'),
    layers.Dense(512, activation='relu'),
    layers.Dropout(0.1),
    layers.Dense(512, activation='relu'),
    layers.Dropout(0.1),
    layers.Dense(6, activation='softmax')

])

# Compiling the model with Adamax Optimizer
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

call = callbacks.EarlyStopping(patience=20, min_delta=0.0001, restore_best_weights=True)

history = model.fit(Xtr, Ytr, validation_data=(xval,yval), batch_size=40, epochs=250, callbacks=[call])

result= model.evaluate(xte,yte)
print("\n정확도 :  %.4f \n\n\n\n" %(result[1] * 100))

training = pd.DataFrame(history.history)
training.loc[:, ['loss', 'val_loss']].plot()
training.loc[:, ['accuracy', 'val_accuracy']].plot()


xhat_idx = np.random.choice(xte.shape[0], 30)
xhat = xte[xhat_idx]
yhat = model.predict(xhat)
for i in range(30):
    print('True : ' + str(np.argmax(yte.iloc[xhat_idx[i]])) + ', Predict : ' + str(np.argmax(yhat[i])))


model.save('fft_model3.h5')


plot_model(model, to_file='model.png', show_shapes=True, )

plt.show()