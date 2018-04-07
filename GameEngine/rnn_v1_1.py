import game_learner as gs
from keras.models import Sequential, Model
from keras.layers import Lambda, Input,Concatenate,Merge, Flatten, LSTM,Conv2D,ConvLSTM2D, Conv2DTranspose, Conv3DTranspose
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

data = gs.load_data()   # All games
# # X, y = gs.preprocess_game_data(data[0])
model = Sequential()

# # model2 = Sequential()
# # rnnmodel = Sequential()
#
train_data = Input(shape=(3,160,160,3) )
LSTM2 = ConvLSTM2D( filters=3,return_sequences=True, kernel_size=(3,3) ) (train_data)
# LSTM3 = LSTM2[:,0,:,:,:]
LSTMa = Lambda(lambda x:x[:,0,:,:,:]) (LSTM2)
LSTMb = Lambda(lambda x:x[:,1,:,:,:]) (LSTM2)
LSTMc = Lambda(lambda x:x[:,2,:,:,:]) (LSTM2)
# LSTM3 = tf.gather(LSTM2, [0,2,3,4])
# LSTM3  = tf.reshape(-1,158,158,3)
LSTM4a = Conv2DTranspose(filters=3, kernel_size=(3,3)) (LSTMa)
LSTM4b = Conv2DTranspose(filters=3, kernel_size=(3,3)) (LSTMb)
LSTM4c = Conv2DTranspose(filters=3, kernel_size=(3,3)) (LSTMc)
# LSTM4maybe = Concatenate(axis=1)([LSTM4a,LSTM4b,LSTM4c])
LSTM4 = Lambda(lambda x :tf.stack([x[0],x[1], x[2]],axis = 1)) ([LSTM4a,LSTM4b,LSTM4c])
model2 = Model(inputs=train_data, outputs=LSTM4)

#
#
# # define LSTM model
# output_lstm_2d = ConvLSTM2D(input_shape=(3,160,160,3), filters=3, kernel_size=(3,3))(train_data)
model.add(ConvLSTM2D(input_shape=(3,160,160,3), filters=3, kernel_size=(3,3)))
# # model.add( Flatten() )
#
# # rnnmodel.add(LSTM(input_shape=(3,160,160,3), filters=3, kernel_size=(3,3)))
# # model.add(ConvLSTM2D(input_shape=(3,160,160,3), filters=3, kernel_size=(3,3)))
# # write custom layer for transformation using action
# # model2.add(Merge([model,rnnmodel]), mode = 'concat')
#
# print(output_lstm_2d.shape)
#
# output_conv2d = Conv2DTranspose(filters=3, kernel_size=(3,3))
model.add(Conv2DTranspose(filters=3, kernel_size=(3,3)))
# # model.add(Dense(...))
#
#
#
#
model.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy'])
model2.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy'])

# model.compile(optimizer='adam',
#               loss={'final_output': 'mean_squared_error'},
#               metrics = ['accuracy'],
#               target_tensors = final_output)
print(model2.summary())
q= 1
for game_data in data:    #1 game point
    if (q%200 == 0):
        inp = input("wana break?")
        if (inp =='y'):
            break
    q+=1
    print(q)
    X, a = gs.preprocess_game_data(game_data,3)
    # y = []
    # for i in range(0, len(X)-1 ):
    #     y.append(X[i+1][2])
    # y = np.array(y)
    y = X[1:]
    X = X[0:-1]
    model2.fit(X, y, epochs=4, batch_size=1, verbose=2)
    #

X, a = gs.preprocess_game_data(game_data,3)
y6 = model.predict(X)
plt.imshow(y6[25])
plt.show()
    # for x in X[0]:
    #     print(len(x))
    # break
    # model.fit(X, y, epochs=10, batch_size=1, verbose=2)
    # model.fit({'train_data': X},
    #           y = y,
    #           epochs = 10, batch_size = 1, verbose = 2)

print("EOF")