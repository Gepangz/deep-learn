import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

keep_prob = 1

def add_layer(inputs, in_size, out_size, activation_function=None):
    Weights = tf.Variable(tf.random_normal([in_size, out_size]))
    biases = tf.Variable(tf.zeros([1, out_size]) + 0.1)
    Wx_plus_b = tf.matmul(inputs, Weights) + biases
    Wx_plus_b = tf.nn.dropout(Wx_plus_b, keep_prob)
    if activation_function is None:
        outputs = Wx_plus_b
    else:
        outputs = activation_function(Wx_plus_b)
    return outputs

# Make up some real data
x_data = np.linspace(-1, 1, 300)[:, np.newaxis]
noise = np.random.normal(0, 0.05, x_data.shape)
y_data = np.square(x_data) - 0.5 + noise

##plt.scatter(x_data, y_data)
##plt.show()

# define placeholder for inputs to network
xs = tf.placeholder(tf.float32, [None, 1])
ys = tf.placeholder(tf.float32, [None, 1])
# add hidden layer
l1 = add_layer(xs, 1, 10, activation_function=tf.nn.relu)
# add output layer
prediction = add_layer(l1, 10, 1, activation_function=None)

# the error between prediciton and real data
loss = tf.reduce_mean(tf.reduce_sum(tf.square(ys-prediction), reduction_indices=[1]))
train_step = tf.train.GradientDescentOptimizer(0.1).minimize(loss)
# important step


if __name__ == '__main__':
    init = tf.initialize_all_variables()
    sess= tf.Session()
    merged = tf.summary.merge_all()
    # summary writer goes in here
    writer = tf.summary.FileWriter("logs/train", sess.graph)
    # test_writer = tf.summary.FileWriter("logs/test", sess.graph)
    sess.run(init)
    # plot the real data
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    ax.scatter(x_data, y_data)
    plt.ion()
    plt.show()


    for i in range(1000):
        # training
        sess.run(train_step, feed_dict={xs: x_data, ys: y_data})
        # merged = tf.summary.merge_all()
        # # summary writer goes in here
        # train_writer = tf.summary.FileWriter("logs/train", sess.graph)
        # test_writer = tf.summary.FileWriter("logs/test", sess.graph)
        if i % 50 == 0:
            # to visualize the result and improvement
            try:
                ax.lines.remove(lines[0])
            except Exception:
                pass
            prediction_value = sess.run(prediction, feed_dict={xs: x_data})
            print(sess.run(loss, feed_dict={xs: x_data, ys: y_data}))
            result = sess.run(merged, feed_dict={xs: x_data, ys: y_data})
            writer.add_summary(result, i)

            # train_result = sess.run(merged, feed_dict={xs: X_train, ys: y_train, keep_prob: 1})
            # test_result = sess.run(merged, feed_dict={xs: X_test, ys: y_test, keep_prob: 1})
            # train_writer.add_summary(train_result, i)
            # test_writer.add_summary(test_result, i)
            # plot the prediction
            lines = ax.plot(x_data, prediction_value, 'r-', lw=5)
            plt.pause(1)