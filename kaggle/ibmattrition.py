import numpy as np
import tensorflow as tf

data = np.load('matrix.npy')

negative = data[np.where(data[:,31] == 0)[0], :]
negative = negative[61:, :]
positive = data[np.where(data[:,31] == 1)[0], :]
positive = positive[1:, :]

np.random.shuffle(negative)
np.random.shuffle(positive)

negative_train = negative[::2,:]
negative_test = negative[1::2,:]
positive_train = positive[::2,:]
positive_test = positive[1::2,:]

train = np.concatenate((negative_train, positive_train), axis=0)
np.random.shuffle(train)
tet = np.concatenate((negative_test, positive_test), axis=0)
np.random.shuffle(tet)

train_input = train[:,0:31]
train_target = train[:,31:33]
test_input = tet[:,0:31]
test_target = tet[:,31:33]

test_positive = sum(1 if x == 1 else 0 for x in test_target[:,0])
test_negative = sum(1 if x == 0 else 0 for x in test_target[:,0])
test_total = test_negative + test_positive
print(test_positive, test_negative, test_total)


batch_size = 32
hidden1 = 30
hidden2 = 10
n_feature = 31
learning_rate = 0.001
n_epoch = 200

X = tf.placeholder(dtype=tf.float32, shape=(batch_size, n_feature))
Y = tf.placeholder(dtype=tf.float32, shape=(batch_size, 2))

w1 = tf.Variable(tf.truncated_normal(shape=(n_feature, hidden1), stddev=0.01, dtype=tf.float32))
b1 = tf.Variable(tf.truncated_normal(shape=(1, hidden1), stddev=0.01, dtype=tf.float32))
y1 = tf.matmul(X, w1) + b1

w2 = tf.Variable(tf.truncated_normal(shape=(hidden1, 2), stddev=0.01, dtype=tf.float32))
b2 = tf.Variable(tf.truncated_normal(shape=(1, 2), stddev=0.01, dtype=tf.float32))
logits = tf.matmul(y1, w2) + b2

# w3 = tf.Variable(tf.truncated_normal(shape=(hidden2, 2), stddev=0.01, dtype=tf.float32))
# b3 = tf.Variable(tf.truncated_normal(shape=(1, 2), stddev=0.01))

# logits = tf.matmul(y2, w3) + b3

loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=Y))

optimizer = tf.train.AdamOptimizer(learning_rate).minimize(loss)

n_batch = test_total // batch_size
with tf.Session() as sess:
    sess.run(tf.initialize_all_variables())
    for i in range(n_epoch):
        total_cost = 0
        for j in range(n_batch):
            X_batch, Y_batch = train_input[batch_size*j:batch_size*(j+1), :], train_target[batch_size*j:batch_size*(j+1),:]
            _, cost = sess.run([optimizer, loss], feed_dict={X: X_batch, Y:Y_batch})
            total_cost += cost
        if i % 10 == 0:
            print('loss epoch {0}: {1}'.format(i+1, total_cost/n_batch))

    #test
    true_positive = 0
    true_negative = 0
    correct_pred = 0
    for j in range(n_batch):
        X_batch, Y_batch = test_input[batch_size * j:batch_size * (j + 1), :], test_target[ batch_size * j:batch_size * (j + 1),:]
        output = sess.run(logits, feed_dict={X: X_batch})
        output = sess.run(tf.nn.softmax(output))
        # print(output[:5,:])
        # print(Y_batch[:5,:])
        true_pred = np.equal(np.argmax(output, 1), np.argmax(Y_batch, 1))
        # accuracy = (tf.cast(true_pred, tf.float32)  # need numpy.count_nonzero(boolarr) :(
        for z in range(len(output)):
            if true_pred[z]:
                correct_pred += 1
                #print(output[z])
                if np.argmax(output[z]) == 0:
                    true_positive += 1
                else: true_negative += 1


    print('correct predictions:', correct_pred / test_total)
    print('true positive:', true_positive/test_positive)
    print('true negative:', true_negative/test_negative)