import game_learner as gs
from keras.models import Model
from keras.layers import Lambda, Input, ConvLSTM2D, Bidirectional, Conv2DTranspose, Dense, Reshape, Dot
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import keras.backend as K

def _np_one_hot(x, n):  # x - no. of examples, # n - Dimension of one hot
    y = np.zeros([len(x), n])
    y[np.arange(len(x)), x] = 1
    return y

def multiplyAction(images, actions):
    images = tf.multiply(actions, images)
    return images

def func(am):
    this_func_list = []
    for i in range(0, nt):
        tmp_output = DeconvLayerShared(am[:, i, :, :, :])
        this_func_list.append(tmp_output)
    return K.stack(this_func_list, axis = 1)

data = gs.load_data()
nt = 8
train_data = Input(shape=(nt, 80, 80, 3))
actions = Input(shape=(nt, 6))
act_reshape = Reshape((nt*6,), input_shape = (nt,6))(actions)
action_FC = Dense(nt*78*78*3, input_shape=(nt*6,))(act_reshape)
LSTM_1 = ConvLSTM2D(filters=3, return_sequences=True, kernel_size=(3, 3))(train_data)
reshape = Reshape((nt*78*78*3,), input_shape = (nt,78,78,3))(LSTM_1)
ActionMultiply = Lambda(lambda x: multiplyAction(x[0], x[1]))([reshape, action_FC])
reshape2 = Reshape((nt, 78, 78, 3), input_shape = (nt*78*78*3,))(ActionMultiply)
DeconvLayerShared = Conv2DTranspose(filters=3, kernel_size=(3, 3))
OUT = Lambda(func)(reshape2) #output_shape=(nt, 80, 80, 3))(reshape2)
print(OUT.shape)
model = Model(inputs=[train_data, actions], outputs=OUT)
model.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy'])
model.summary()

q = 1
for i in range(0, len(data)):  # 1 game point
    if q % 20 == 0:
        break
    game_data = data[i]
    q += 1
    print(q)
    X, a, y = gs.preprocess_game_data(game_data, nt)
    print(y.shape)
    print(a.shape)
    L = []
    for j in range(0, nt):
        action = a[:, j]
        L.append(_np_one_hot(action, 6))
    L = np.array(L)
    L = np.transpose(L, (1, 0, 2))
    print(X.shape, L.shape, y.shape)
    model.fit([X, L], y, epochs=10, batch_size=1, verbose=2)
    print(X.shape, L.shape, y.shape)
    model.save_weights('rnn7_weights_rgb.h5')


X, a, y = gs.preprocess_game_data(data[i+1], nt)
L = []
for j in range(0, nt):
    action = a[:, j]
    L.append(_np_one_hot(action, 6))
L = np.array(L)
L = np.transpose(L, (1, 0, 2))
print(X.shape, L.shape, y.shape)
preds = model.predict([X,L])
preds = preds + np.load('meanrgb.npy')
print(preds.shape)
plt.imsave('X.png', X[0, 0, :, :, :])
for j in range(0, preds.shape[0]):
    plt.imsave('./Iter_i' + str(i) +'j' +  '%07d' % (j) + '.png', preds[j][1][:, :, :])