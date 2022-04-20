#ライブラリのインポート
import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, MaxPooling2D, BatchNormalization
import copy
from IPython import display
from sklearn.model_selection import train_test_split

#TensorFlowに入っているcifar10をインポート
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()

y_train = np.squeeze(y_train)
y_test = np.squeeze(y_test)

cat_idx = 3 #cifar10内での猫のラベル
dog_idx = 5 #cifar10内での犬のラベル

x_cat_train = x_train[np.where(y_train==cat_idx,True,False)]
x_cat_test = x_test[np.where(y_test==cat_idx,True,False)]

x_dog_train = x_train[np.where(y_train==dog_idx,True,False)]
x_dog_test = x_test[np.where(y_test==dog_idx,True,False)]

'''
def plot_imgs(input_imgs):
    plt.figure(figsize=(10,10))
    plt_idx = 1
    for i in range(36):
        plt.subplot(6,6,i+1)
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        plt.imshow(np.squeeze(input_imgs[i]), cmap="gray", vmin=0, vmax=1)
    plt.show()
'''

#plot_imgs(x_dog_test)

#plot_imgs(x_cat_train)

x_train = np.concatenate((x_cat_train, x_dog_train))
x_test = np.concatenate((x_cat_test, x_dog_test))

y_train = np.concatenate((np.zeros(x_cat_train.shape[0]),np.ones(x_dog_train.shape[0])))
y_test = np.concatenate((np.zeros(x_cat_test.shape[0]),np.ones(x_dog_test.shape[0])))

x_train, x_valid, y_train, y_valid = train_test_split(x_train,y_train,test_size=0.33, random_state=42)


#CNNの構築
model = Sequential([
    Conv2D(16, 3, padding='same', activation='relu', input_shape=(32, 32 ,3)),
    MaxPooling2D(),
    Conv2D(32, 3, padding='same', activation='relu'),
    MaxPooling2D(),
    Conv2D(64, 3, padding='same', activation='relu'),
    MaxPooling2D(),
    Flatten(),
    Dense(512, activation='relu'),
    Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

#fit関数による学習
history = model.fit(x_train, y_train, epochs=50, batch_size=32,validation_data=(x_valid, y_valid))


history = history.history

#学習時の誤差の推移グラフの出力
plt.plot(np.arange(len(history["loss"])),history["loss"], label="loss")
plt.plot(np.arange(len(history["val_loss"])),history["val_loss"], label="val_loss")
plt.legend()
plt.show()

#学習時の正答率の推移グラフの出力
plt.plot(np.arange(len(history["accuracy"])), history["accuracy"], label="accuracy")
plt.plot(np.arange(len(history["val_accuracy"])), history["val_accuracy"], label="val_accuracy")
plt.legend()


print ("実行成功")