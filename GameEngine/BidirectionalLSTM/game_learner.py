# import _pickle as pickle
import pickle
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

def load_data():
	return pickle.load(open('dataset.p','rb'))

def preprocess_game_data(game_data,inum):
	game_data = np.array(game_data)
	image_action_pairs = np.apply_along_axis(preprocess_single_sample, axis=1, arr = np.expand_dims(np.array(game_data), axis=1))
	images = image_action_pairs[:, 0]
	actions = image_action_pairs[:, 1]

	# img = Image.fromarray(images[3], 'RGB')
	# img.show()
	# img = Image.fromarray(images[4], 'RGB')
	# img.show()

	# images_copy = np.copy(images)
	# for i in range(1, len(images)):
	# 	images_copy[i] = images[i] - images[i - 1]
	# images = images_copy

	# img = Image.fromarray(images[4], 'RGB')
	# img.show()
	# print(np.unique(images[4]))

	num_images_in_each_sample = inum
	images_list = images.tolist()
	X = list(map(image_to_image_list, [images_list]*(len(images_list)-(inum-1)), range(0, len(images_list)-(inum-1)), [inum]*(len(images_list)-(inum-1))))
	# a = actions[inum - 1:].tolist()

	# X = list(map(image_to_image_list, [images_list] * (len(images_list) - inum),
	#              range(0, len(images_list) - inum), [inum] * (len(images_list) - inum)))
	# X = list(map(image_to_image_list, [images_list] * (len(images_list) - inum), range(0, len(images_list) - inum), [inum] * (len(images_list) - inum)))
	# y = np.expand_dims(images[inum:], axis=1)
	# y = list(map(lambda image: image.tolist(), images_list[inum:]))
	y = X[1:]
	X = X[:-1]
	a = actions[:-1].tolist()
	a_formatted = []
	for i in range(0, len(a)-inum+1):
		a_formatted.append(a[i:i+inum])

	# return np.array(X), np.expand_dims(a, axis=1), np.array(y)
	# return np.array(X), np.expand_dims(a_formatted, axis=2), np.array(y)
	return np.array(X), np.array(a_formatted), np.array(y)

def image_to_image_list(images, i, out_size):
	if i > len(images) - out_size:
		return
	# return np.concatenate(images[i:i+out_size], axis=2)#,images[i+out_size]
	# return np.stack(images[i:i+out_size],axis=2)
	return images[i:i + out_size]

def preprocess_single_sample(raw):
	raw = np.array(raw[0]).astype(np.int)
	image = np.zeros((80, 80, 3),dtype=np.uint8)
	# background
	image[:, :, 0] = 144
	image[:, :, 1] = 72
	image[:, :, 2] = 17

	for i in range(0, 3):
		flat_image = image[:, :, i].ravel()
		flat_image[raw[:-1]] = 1
		image[:, :, i] = np.reshape(flat_image, (80, 80))

	left_paddle_indices = np.where(image[:, :10, :] == 1)
	# image[left_paddle_indices[0],left_paddle_indices[1],0] = 213
	# image[left_paddle_indices[0], left_paddle_indices[1], 1] = 130
	# image[left_paddle_indices[0], left_paddle_indices[1], 2] = 74
	image[left_paddle_indices[0], left_paddle_indices[1], 0] = 1
	image[left_paddle_indices[0], left_paddle_indices[1], 1] = 186
	image[left_paddle_indices[0], left_paddle_indices[1], 2] = 92
	# image[left_paddle_indices[0], left_paddle_indices[1], 0] = 255
	# image[left_paddle_indices[0], left_paddle_indices[1], 1] = 255
	# image[left_paddle_indices[0], left_paddle_indices[1], 2] = 255

	ball_indices = np.where(image[:, :70, :] == 1)
	# print(ball_indices)
	# image[ball_indices] = 236
	image[ball_indices] = 255

	right_paddle_indices = np.where(image == 1)
	image[right_paddle_indices[0], right_paddle_indices[1], 0] = 1
	image[right_paddle_indices[0], right_paddle_indices[1], 1] = 186
	image[right_paddle_indices[0], right_paddle_indices[1], 2] = 92
	# image[right_paddle_indices[0], right_paddle_indices[1], 0] = 255
	# image[right_paddle_indices[0], right_paddle_indices[1], 1] = 255
	# image[right_paddle_indices[0], right_paddle_indices[1], 2] = 255

	# image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	# image = np.expand_dims(image, axis=2)
	# image = Image.fromarray(image, 'RGB').convert('L')
	# image = np.expand_dims(np.array(image), axis=2)


	# img = Image.fromarray(image, 'RGB')
	# img.show()

	# return image.repeat(2,axis=0).repeat(2,axis=1), raw[-1]
	return np.array(image)/255.0, raw[-1]
	# mean_subtracted = (np.array(image) / 255.0) - np.load('mean.npy')
	# return mean_subtracted, raw[-1]
	# return np.expand_dims(mean_subtracted, axis=2), raw[-1]

if __name__ == "__main__":
	data = load_data()
	print(data.shape)
	inum = 5
	X, a, y = preprocess_game_data(data[0], inum)
	print(X.shape)
	print(a.shape)
	print(y.shape)
	#
	# temp = np.zeros((data.shape[0],80,80))
	# for i in range(0, data.shape[0]):
	# 	game_data = np.array(data[i])
	# 	image_action_pairs = np.apply_along_axis(preprocess_single_sample, axis=1,
	# 											 arr=np.expand_dims(np.array(game_data), axis=1))
	# 	images = image_action_pairs[:, 0]
	# 	# print(np.mean(images, axis=0).shape)
	# 	temp[i] = np.mean(images, axis=0)
	#
	# mean = np.mean(temp, axis=0)
	#
	# # print(X.shape)
	# # print(y.shape)
	#
	# # plt.imshow(temp[0], cmap='gray', vmin=0, vmax=1)
	# # plt.show()
	#
	# np.save('mean.npy', mean)


# img = Image.fromarray(X[0, 0], 'RGB')
	# img.show()
	# img = Image.fromarray(X[0, 0] + X[0,1], 'RGB')
	# img.show()
	# img = Image.fromarray(X[0, 0] + X[0,1] + X[0,2], 'RGB')
	# img.show()
	# img = Image.fromarray(X[0, 0] + X[0,1] + X[0,2] + X[0,3], 'RGB')
	# img.show()
	# img = Image.fromarray(X[0, 0] + X[0,1] + X[0,2] + X[0,3] + X[0,4], 'RGB')
	# img.show()
