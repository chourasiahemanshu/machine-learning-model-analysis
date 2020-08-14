import torch
import torch.nn as nn
import pickle
import numpy as np
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
from models.decision_tree_classifier import DTClassifier
from models.model import Model

torch.manual_seed(0)

MODEL_PATH = '../cifar_net.pth'
CIFAR_DATA_PATH = './cfar10/'

BATCH_SIZE = 4
MOMENTUM = 0.9
LEARNING_RATE = 0.001
EPOCH = 10
ACTIVATION_MAP_CLASS = 6

class SimpleNet(nn.Module):
	def __init__(self):
		super(SimpleNet, self).__init__()
		self.conv1 = nn.Conv2d(3, 15, 3)
		self.conv2 = nn.Conv2d(15, 25, 3)
		self.pool = nn.MaxPool2d(2, 1)
		self.pool2 = nn.MaxPool2d(2, 2)
		self.conv3 = nn.Conv2d(25, 50, 3)
		self.conv4 = nn.Conv2d(50, 50, 5)
		self.fc1 = nn.Linear(50 * 4 * 4, 120)
		self.fc2 = nn.Linear(120, 84)
		self.fc3 = nn.Linear(84, 10)

	def forward(self, x):
		x = self.pool(F.relu(self.conv1(x)))
		x = self.pool(F.relu(self.conv2(x)))
		x = self.pool2(F.relu(self.conv3(x)))
		x = self.pool2(F.relu(self.conv4(x)))
		x = x.view(-1, 50 * 4 * 4)
		x = F.relu(self.fc1(x))
		x = F.relu(self.fc2(x))
		x = self.fc3(x)
		return x

class CifarModel():
	def __init__(self):
		print("""
    		**********************
    		CIFAR DATASET
    		**********************
    	""")


	def load_cfar10_batch(self, name):
		with open('{}{}'.format(CIFAR_DATA_PATH, name), 'rb') as file:
			batch = pickle.load(file, encoding='bytes')
			features = batch[b'data'] #.reshape((len(batch[b'data']), 3, 32, 32)).transpose(0, 2, 3, 1)
			labels = batch[b'labels']
		return features, labels


	def load_cfar10_data(self):
		x1, y1 = self.load_cfar10_batch('data_batch_1')
		x2, y2 = self.load_cfar10_batch('data_batch_2')
		x3, y3 = self.load_cfar10_batch('data_batch_3')
		x4, y4 = self.load_cfar10_batch('data_batch_4')

		self.X_train = np.concatenate((x1,x2,x3,x4))
		self.y_train = y1+y2+y3+y4
		self.X_test, self.y_test = self.load_cfar10_batch('test_batch')

	def normalize_numpy(self, x):
		min_val = np.min(x)
		max_val = np.max(x)
		x = (x-min_val) / (max_val-min_val)
		return x

	def train_and_test_dt(self):
		dt_classifier = DTClassifier('./cfar10/data_batch')
		model = Model(model_type=dt_classifier)
		model.perform_experiment_for_cifar(self.X_train, self.X_test, self.y_train, self.y_test)
		model.model_type.plot_and_save_tree(max_depth=500)

	def normalize_data(self):
		self.X_train_scaled = self.normalize_numpy(self.X_train)
		self.X_test_scaled = self.normalize_numpy(self.X_test)

	def reshape_and_convert_to_tensor(self):
		self.X_train_scaled_tensor = self.X_train_scaled.reshape(self.X_train_scaled.shape[0], 3, 32, 32)
		self.X_test_scaled_tensor = self.X_test_scaled.reshape(self.X_test_scaled.shape[0], 3, 32, 32)
		self.X_train_scaled_tensor = torch.from_numpy(self.X_train_scaled_tensor).float()
		self.X_test_scaled_tensor = torch.from_numpy(self.X_test_scaled_tensor).float()

		self.y_train_tensor = torch.tensor(np.array(self.y_train).astype(np.int64))
		self.y_test_tensor = torch.tensor(np.array(self.y_test).astype(np.int64))

	def load_and_normalize_data(self):
		self.load_cfar10_data()
		self.normalize_data()

	def train_and_save_cnn(self):
		self.reshape_and_convert_to_tensor()
		net = SimpleNet()
		criterion = nn.CrossEntropyLoss()
		optimizer = optim.SGD(net.parameters(), lr=LEARNING_RATE, momentum=MOMENTUM)
		batch_size = BATCH_SIZE

		print("starting cnn training ===>")

		for epoch in range(EPOCH):  # loop over the dataset multiple times
			for i in range(0, len(self.X_train_scaled_tensor), batch_size):
				inputs = self.X_train_scaled_tensor[i:i+batch_size]
				labels = self.y_train_tensor[i:i+batch_size]

				# zero the parameter gradients
				optimizer.zero_grad()

				# forward + backward + optimize
				outputs = net(inputs)
				loss = criterion(outputs, labels)
				loss.backward()
				optimizer.step()

			print("Epoch {} final minibatch had loss {}".format(epoch, loss.item()))

		print('Finished cnn Training, Saving model')
		torch.save(net.state_dict(), MODEL_PATH)

	def test_cnn(self):
		print("starting cnn testing ====>")
		net = SimpleNet()
		net.load_state_dict(torch.load(MODEL_PATH))

		classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
		class_correct = [0. for i in range(10)]
		class_total = [0. for i in range(10)]
		correct = 0
		total = 0
		with torch.no_grad():
			for i in range(self.X_test_scaled_tensor.shape[0]):
				image =  self.X_test_scaled_tensor[i].reshape(1, 3,32,32)
				label = self.y_test_tensor[i]
				outputs = net(image)
				_, predicted = torch.max(outputs.data, 1)
				total += 1
				class_total[label] += 1
				if predicted == label:
					correct += 1
					class_correct[label] += 1

		print('Accuracy of the network on the 10000 test images: %d %%' % (100 * correct / total))
		for i in range(10):
			print('Accuracy of %5s : %2d %%' % (classes[i], 100 * class_correct[i] / class_total[i]))

	def gray_conversion(self, image):
		gray_image = np.dot(image, [0.299, 0.587, 0.144])
		return gray_image

	def activation_maximization_cnn(self):
		print("starting cnn activation maximization ====>")

		net = SimpleNet()
		net.load_state_dict(torch.load(MODEL_PATH))

		classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

		X = torch.zeros([1, 3, 32,32], requires_grad=True).float()
		criterion = nn.CrossEntropyLoss()
		for idx, i in enumerate(range(5000)):
			y = net(X)
			_, predicted = torch.max(y.data, 1)

			if idx == 0:
				print("creating activation map for", classes[ACTIVATION_MAP_CLASS])

			loss = criterion(y, torch.tensor(np.array(ACTIVATION_MAP_CLASS).astype(np.int64)).view(1))
			loss.backward()

			X.data += 0.6*X.grad.data
			loss.data.zero_()


		activation_image = X.detach().numpy().reshape(3,32,32).transpose(1,2,0)
		activation_image = self.gray_conversion(activation_image)
		plt.imsave('{}_1.png'.format(classes[ACTIVATION_MAP_CLASS]), activation_image, cmap = plt.get_cmap('gray'))
