import game_learner as gs
from keras.models import Sequential
from keras.layers import Merge, Flatten, LSTM,Conv2D,ConvLSTM2D, Conv2DTranspose
import numpy as np

from tqdm import tqdm

data = gs.load_data()   # All games
# X, y = gs.preprocess_game_data(data[0])
model = Sequential()
model2 = Sequential()
rnnmodel = Sequential()
# define LSTM model
model.add(Conv2D(input_shape=(3,160,160,3), filters=3, kernel_size=(3,3)))
model.add( Flatten() )

rnnmodel.add(LSTM(input_shape=(3,160,160,3), filters=3, kernel_size=(3,3)))
# model.add(ConvLSTM2D(input_shape=(3,160,160,3), filters=3, kernel_size=(3,3)))
# write custom layer for transformation using action
model2.add(Merge([model,rnnmodel]), mode = 'concat')
model2.add(Conv2DTranspose(filters=3, kernel_size=(3,3)))
# model.add(Dense(...))
model2.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
for game_data in data:    #1 game point
    X, a = gs.preprocess_game_data(game_data,3)
    y = []
    for i in range(0, len(X)-1 ):
        y.append(X[i+1][2])
    y = np.array(y)
    X = X[0:-1]
    print ("EOF")
    model2.fit(X, y, epochs=10, batch_size=1, verbose=2)

print("EOF")