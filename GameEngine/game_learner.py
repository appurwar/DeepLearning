import _pickle as pickle
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM, ConvLSTM2D, Conv2DTranspose
from keras.utils import np_utils
from keras.layers import Conv2D, MaxPool2D, TimeDistributed

from PIL import Image

def load_data():
    return pickle.load(open('dataset.p','rb'))

def preprocess_game_data(game_data,inum):
    game_data = np.array(game_data)
    # print(type(game_data))
    # print(type(game_data[0]))
    # print(type(game_data[0][0]))
    image_action_pairs = np.apply_along_axis(preprocess_single_sample, axis=1, arr=np.expand_dims(np.array(game_data), axis=1))
    images = image_action_pairs[:, 0]
    actions = image_action_pairs[:, 1]

    num_images_in_each_sample = inum
    images_list = images.tolist()
    X = list(map(image_to_image_list, [images_list]*(len(images_list)-2), range(0, len(images_list)-2), [num_images_in_each_sample]*(len(images_list)-2)))
    y = actions[num_images_in_each_sample - 1:].tolist()

    print("X shape: {}".format(np.array(X).shape))
    print("y shape: {}".format(np.array(y).shape))
    return np.array(X), np.expand_dims(y, axis=1)

def image_to_image_list(images, i, out_size):
    if i > len(images) - out_size:
        return
    return images[i: i + 3]

def preprocess_single_sample(raw):
    # raw = np.array(raw)
    # print(raw)
    raw = np.array(raw[0]).astype(np.int)
    # print(raw.dtype)
    image = np.zeros((80, 80, 3),dtype=np.uint8)
    # background
    image[:, :, 0] = 144
    image[:, :, 1] = 72
    image[:, :, 2] = 17

    for i in range(0, 3):
        flat_image = image[:, :, i].ravel()
        # print("Raw: ", raw)
        # print(flat_image.shape)
        flat_image[raw[:-1]] = 1
        image[:, :, i] = np.reshape(flat_image, (80, 80))

    left_paddle_indices = np.where(image[:, :10, :] == 1)
    # print(left_paddle_indices)
    image[left_paddle_indices[0],left_paddle_indices[1],0] = 213
    image[left_paddle_indices[0], left_paddle_indices[1], 1] = 130
    image[left_paddle_indices[0], left_paddle_indices[1], 2] = 74

    ball_indices = np.where(image[:, :70, :] == 1)
    # print(ball_indices)
    image[ball_indices] = 236

    right_paddle_indices = np.where(image == 1)
    image[right_paddle_indices[0], right_paddle_indices[1], 0] = 1
    image[right_paddle_indices[0], right_paddle_indices[1], 1] = 186
    image[right_paddle_indices[0], right_paddle_indices[1], 2] = 92

    # img = Image.fromarray(image, 'RGB')
    # img.show()

    return image.repeat(2,axis=0).repeat(2,axis=1).tolist(), raw[-1]

data = load_data()
# img, ac = preprocess_single_sample(data[40][50])
# Image.fromarray(img, 'RGB').show()
# X, y = preprocess_game_data(data[0])

model = Sequential()
# define LSTM model
model.add(ConvLSTM2D(input_shape=(3,160,160,3), filters=3, kernel_size=(3,3)))
# write custom layer for transformation using action
model.add(Conv2DTranspose(filters=3, kernel_size=(3,3)))
# model.add(Dense(...))

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# for game_data in data:
#     X, y = preprocess_game_data(game_data)
#     print(type(X))
#     print(type(X[0]))
#     print(type(X[0][0]))
#     model.fit(X, y, epochs=500, batch_size=1, verbose=2)


# # fix random seed for reproducibility
# np.random.seed(7)
#
# raw_data = pickle.load(open('dataset.p', 'r'))
#
# timesteps  = 3
#
# # TODO: normalize inputs between 0 to 1
#
# # create and fit the model
# model = Sequential()
# model.add(LSTM(32, input_shape=(X.shape[1], X.shape[2])))
# model.add(Dense(y.shape[1], activation='softmax'))
# model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
# model.fit(X, y, epochs=500, batch_size=1, verbose=2)
# # summarize performance of the model
# scores = model.evaluate(X, y, verbose=0)
# print("Model Accuracy: %.2f%%" % (scores[1]*100))