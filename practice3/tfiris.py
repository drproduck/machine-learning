import tensorflow as tf
import numpy as np

n_epoches = 2000
learning_rate = 0.01


data = np.loadtxt('iris_proc.data', delimiter=',')

np.random.shuffle(data)

target = np.zeros((np.shape(data)[0], 3))
indices = np.where(data[:,4]==0)
target[indices,0] = 1
indices = np.where(data[:,4]==1)
target[indices,1] = 1
indices = np.where(data[:,4]==2)
target[indices,2] = 1

train = data[::2, :4]
traint = target[::2, :]

tet = data[1::2, :4]
tett = target[1::2, :]

X = tf.placeholder(tf.float32, (np.shape(train)[0], 4), name='X_placeholder')
Y = tf.placeholder(tf.float32, (np.shape(train)[0], 3), name='Y_placeholder')

w = tf.Variable(tf.random_normal(shape=(4, 5), stddev=0.01), name='weights1')
w1 = tf.Variable(tf.random_normal(shape=(5,3), stddev=0.01), name='weights2')
b = tf.Variable(tf.zeros(shape=(1, 5)), name='bias1')
b1 = tf.Variable(tf.zeros(shape=(1, 3)), name='bias2')

logits = tf.matmul(tf.matmul(X, w) + b, w1) + b1

entropy = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=Y, name='loss')
loss = tf.reduce_mean(entropy)

optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)

loss1 = 1000
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for i in range(n_epoches):
            _, loss_value = sess.run([optimizer, loss], feed_dict={X:train, Y:traint})
            if loss_value > loss1: break
            loss1 = loss_value
            print('loss epoch {0}: {1}'.format(i+1, loss_value))

    print('Finished!')

    logits_test = sess.run(logits, feed_dict={X: tet})
    print(np.concatenate((logits_test, tett), axis=1))
    correct_preds = sess.run(tf.reduce_sum(tf.cast(tf.equal(tf.arg_max(logits_test, 1), tf.argmax(tett, 1)), dtype=tf.float32)))
    print('Accuracy {0}'.format(correct_preds / np.shape(tett)[0]))



