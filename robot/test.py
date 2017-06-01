import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt


INPUT_SIZE = 8
OUTPUT_SIZE = 5
In_data = np.eye(5000, INPUT_SIZE)
Out_data = np.eye(5000, OUTPUT_SIZE)
##构建输入数据beg
def readData(infile, outfile):
	with open(infile, 'r') as f:
		index = 0
		for line in f.readlines():
			inStr = line.strip()
			inSet = list(map(np.float32, inStr.split(' ')))
			In_data[index] = inSet
			index += 1

	with open(outfile, 'r') as f:
		index = 0
		for line in f.readlines():
			outStr = line.strip()
			outSet = list(map(np.float32, outStr.split(' ')))
			Out_data[index] = outSet
			index += 1
readData('./data/input.txt', './data/output.txt')
##构建数据end


# Make up some real data
# x_data = np.linspace(-1, 1, 300)[:, np.newaxis]
# noise = np.random.normal(0, 0.05, x_data.shape)
# y_data = np.square(x_data) - 0.5 + noise
x_data = np.linspace(-1, 1, 300)[:, np.newaxis]
noise = np.random.normal(0, 0.05, x_data.shape)
y_data = np.square(x_data) - 0.5 + noise

In_size = 1
Out_size = 1
Neural_size = 10

class DNN(object):
	def __init__(self, input_size, output_size, neural_size):
		self.input_size = input_size
		self.output_size = output_size
		self.neurals = neural_size

		with tf.name_scope('inputs'):
			self.xs = tf.placeholder(tf.float32, [None, input_size], name='xs')
			self.ys = tf.placeholder(tf.float32, [None, output_size], name='ys')
		with tf.variable_scope('in_hidden'):
			self.l1 = self.add_input_layer(self.xs, self.input_size, self.neurals, activation_function=tf.nn.relu)
		with tf.variable_scope('out_hidden'):
			self.prediction = self.add_output_layer(self.l1, self.neurals, self.output_size, activation_function=None)
		with tf.name_scope('cost'):
			self.cost = self.compute_cost()
		with tf.name_scope('train'):
			self.train_step = tf.train.GradientDescentOptimizer(0.1).minimize(self.cost)

	def add_input_layer(self, inputs, in_size, out_size, activation_function=None):
		
		Weights = tf.Variable(tf.random_normal([in_size, out_size]))
		biases = tf.Variable(tf.zeros([1, out_size]) + 0.1)

		Wx_plus_b = tf.matmul(inputs, Weights) + biases
		if activation_function is None:
			outputs = Wx_plus_b
		else:
			outputs = activation_function(Wx_plus_b)
		return outputs
	def add_output_layer(self, inputs, in_size, out_size, activation_function=None):
	    Weights = tf.Variable(tf.random_normal([in_size, out_size]))
	    biases = tf.Variable(tf.zeros([1, out_size]) + 0.1)
	    Wx_plus_b = tf.matmul(inputs, Weights) + biases
	    if activation_function is None:
	        outputs = Wx_plus_b
	    else:
	        outputs = activation_function(Wx_plus_b)
	    return outputs
	def compute_cost(self):
		return tf.reduce_mean(tf.reduce_sum(tf.square(self.ys-self.prediction), reduction_indices=[1]))




if __name__ == '__main__':
	with tf.Session() as sess:
		model = DNN(In_size, Out_size, Neural_size)
		init = tf.initialize_all_variables()
		sess.run(init)
		fig = plt.figure()
		ax = fig.add_subplot(1,1,1)
		ax.scatter(x_data, y_data)
		plt.ion()
		plt.show()
		for i in range(1000):
			# training
			sess.run(model.train_step, feed_dict={model.xs: x_data, model.ys: y_data})
			if i % 50 == 0:
			# to visualize the result and improvement
				try:
					ax.lines.remove(lines[0])
				except Exception:
					pass
				prediction_value = sess.run(model.prediction, feed_dict={model.xs: x_data})
				print(prediction_value)
				# plot the prediction
				lines = ax.plot(x_data, prediction_value, 'r-', lw=5)
				plt.pause(1)