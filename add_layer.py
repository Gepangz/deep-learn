import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

def add_layer(inputs, in_size, out_size, activation_funcition=None):
	with tf.name_scope('layer'):
		with tf.name_scope('Weights'):
			Weights = tf.Variable(tf.random_normal([in_size, out_size]), name='W')
		with tf.name_scope('biases'):
		biases = tf.Variable(tf.zeros([1, out_size]) + 0.1)
		Wx_plus_b = tf.matmul(inputs, Weights) + biases
		if activation_funcition is None:
			outputs = Wx_plus_b
		else:
			outputs = activation_funcition(Wx_plus_b)
		return outputs

x_data = np.linspace(-1, 1, 300)[:, np.newaxis]
noise = np.random.normal(0, 0.05, x_data.shape)
y_data = np.square(x_data) - 0.5 + noise


with tf.name_scope('inputs'):
	xs = tf.placeholder(tf.float32, [None, 1], name='x_input')
	ys = tf.placeholder(tf.float32, [None, 1], name='y_input')

l1 = add_layer(xs, 1, 10, activation_funcition=tf.nn.relu)
predition = add_layer(l1, 10, 1, activation_funcition=None)

loss = tf.reduce_mean(tf.reduce_sum(tf.square(ys - predition), reduction_indices=[1]));

train_step = tf.train.GradientDescentOptimizer(0.1).minimize(loss)

init = tf.initialize_all_variables()

sess = tf.Session()
sess.run(init)

fig = plt.figure()
ax = fig.add_subplot(1,1,1)
ax.scatter(x_data, y_data)
plt.ion()
plt.show()

for i in range(1000):
	sess.run(train_step, feed_dict={xs: x_data, ys: y_data})
	if i % 50 == 0:
		try:
			ax.lines.remove(lines[0])
		except Exception as e:
			pass
		#print(sess.run(loss, feed_dict={xs: x_data, ys: y_data}))
		predition_value = sess.run(predition, feed_dict={xs:x_data})
		lines = ax.plot(x_data, predition_value, 'r-', lw=5)
		plt.pause(0.1)
