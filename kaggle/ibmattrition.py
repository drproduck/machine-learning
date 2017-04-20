import numpy as np
import tensorflow as tf

data = np.load('matrix.npy')

negative = data[np.where(data[:,31] == 0)[0], :]
negative = negative[61:, :]
total_negative = len(negative)
print(total_negative)
positive = data[np.where(data[:,31] == 1)[0], :]
positive = positive[1:, :]
total_positive = len(positive)
print(total_positive)
total  = total_negative + total_positive
print(total)

np.random.shuffle(negative)
np.random.shuffle(positive)

negative_train = negative[::2,:]
negative_test = negative[1::2,:]
positive_train = positive[::2,:]
positive_test = positive[1::2,:]

train = np.concatenate((negative_train, positive_train), axis=0)
print(len(train))
tet = np.concatenate((negative_test, positive_test), axis=0)
print(len(tet))

train_input = train[:,0:31]
train_target = train[:,31:32]
test_input = tet[:,0:31]
test_target = tet[:,31:32]

batch_size = 64
hidden1 = 60
n_feature = 31
learning_rate = 0.001
n_epoch = 1000

X = tf.placeholder(dtype=tf.float64, shape=(batch_size, n_feature))
Y = tf.placeholder(dtype=tf.float64, shape=(batch_size, 1))

w1 = tf.Variable(tf.truncated_normal(shape=(n_feature, hidden1), stddev=0.01, dtype=tf.float64))
b1 = tf.Variable(tf.truncated_normal(shape=(1, hidden1), stddev=0.01, dtype=tf.float64))

w2 = tf.Variable(tf.truncated_normal(shape=(hidden1, 1), stddev=0.01, dtype=tf.float64))
b2 = tf.Variable(tf.truncated_normal(shape=(1, 1), stddev=0.01, dtype=tf.float64))

logits = tf.matmul(tf.matmul(X, w1) + b1, w2) + b2

loss = tf.reduce_mean(tf.nn .weighted_cross_entropy_with_logits(targets=Y, logits=logits, pos_weight=1.5))

optimizer = tf.train.AdamOptimizer(learning_rate).minimize(loss)

n_batch = 11
with tf.Session() as sess:
    sess.run(tf.initialize_all_variables())
    for i in range(n_epoch):
        total_cost = 0
        for j in range(n_batch):
            X_batch, Y_batch = train_input[batch_size*j:batch_size*(j+1), :], train_target[batch_size*j:batch_size*(j+1),:]
            _, cost = sess.run([optimizer, loss], feed_dict={X: X_batch, Y:Y_batch})
            total_cost += cost
        print('loss epoch {0}: {1}'.format(i+1, total_cost/n_batch))

    #test
    true_positive = 0
    true_negative = 0
    correct_pred = 0
    for j in range(n_batch):
        X_batch, Y_batch = test_input[batch_size * j:batch_size * (j + 1), :], test_target[ batch_size * j:batch_size * (j + 1),:]
        output = sess.run(tf.nn.sigmoid(logits), feed_dict={X: X_batch})
        for k in range(len(X_batch)):
            if abs(output[k] - Y_batch[k]) < 0.01:
                correct_pred += 1
                if Y_batch[k] == 1: true_positive += 1
                else: true_negative += 1

    print('correct predictions:', correct_pred / total)
    print('true positive:', true_positive/total_positive)
    print('true negative:', true_negative/total_negative)