import tensorflow as tf
import numpy as np
import pandas as pd

batch_size = 150
hidden1 = 30
hidden2 = 30
learning_rate = 0.01
num_epochs = 30
num_features = 30
dropout = 1

fraud = np.load('fraud.npy')
normal = np.load('normal.npy')
np.random.shuffle(fraud)
np.random.shuffle(normal)

normal = normal[215:,:]
print(len(normal))

fraud_train = fraud[::2, :]
fraud_test = fraud[1::2, :]
normal_train = normal[1::2, :]
normal_test = normal[::2, :]

train = np.concatenate((fraud_train, normal_train), axis=0)
tet = np.concatenate((normal_test, fraud_test), axis=0)

np.random.shuffle(train)
np.random.shuffle(tet)

train_input = train[:,:num_features]
train_target = train[:, num_features:num_features+1]
test_input = tet[:,:num_features]
test_target = tet[:, num_features:num_features+1]
test_positive = sum(1 if x == 1 else 0 for x in test_target)
test_negative = len(test_target) - test_positive
print('test positive', test_positive)
print('test negative', test_negative)

X = tf.placeholder(dtype=tf.float32, shape=(batch_size, num_features))
Y = tf.placeholder(dtype=tf.float32, shape=(batch_size, 1))

w1 = tf.Variable(tf.truncated_normal((num_features, hidden1), stddev=0.01))
b1 = tf.Variable(tf.zeros((1, hidden1)))
y1 = tf.matmul(X,w1) + b1

w2 = tf.Variable(tf.truncated_normal((hidden1, hidden2), stddev=0.01))
b2 = tf.Variable(tf.zeros((1,hidden2)))
y2 = tf.matmul(y1, w2) + b2

w3 = tf.Variable(tf.truncated_normal((hidden2, 1), stddev=0.01))
b3 = tf.Variable(tf.zeros((1, 1)))
logits = tf.matmul(y2, w3) +  b3

loss = tf.reduce_mean(tf.nn.weighted_cross_entropy_with_logits(targets=Y, logits=logits, pos_weight=30))
#loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=Y, logits=logits))

optimizer = tf.train.AdamOptimizer(learning_rate).minimize(loss)

n_batches = len(train_input) // batch_size
print('number of batches:',n_batches)
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    early_stopping_score = 301
    for epoch_step in range(num_epochs):
        total_loss = 0
        for batch_step in range(n_batches):
            X_batch, Y_batch = train_input[batch_size*batch_step:batch_size*(batch_step+1), :], train_target[batch_size*batch_step:batch_size*(batch_step+1), :]
            _, batch_loss, output = sess.run([optimizer, loss, logits], feed_dict={X: X_batch, Y: Y_batch})
            total_loss += batch_loss
        print('loss epoch {0}: {1}'.format(epoch_step+1, total_loss / n_batches))
        # logits_batch_fraud = sess.run(tf.nn.sigmoid(logits), feed_dict={X: fraud_test[:150, :30]})
        # logits_batch_normal = sess.run(tf.nn.sigmoid(logits), feed_dict={X: normal_test[:150, :30]})
        # test_batch_fraud = fraud_test[:150, 30:31]
        # test_batch_normal = normal_test[:150, 30:31]
        #
        # correct_preds = 0
        # for j in range(len(test_batch_fraud)):
        #     if np.abs(logits_batch_fraud[j] - test_batch_fraud[j]) < 0.01:
        #         correct_preds += 1
        #     if np.abs(logits_batch_normal[j] - test_batch_normal[j]) < 0.01:
        #         correct_preds += 1
        # if correct_preds > 270:
        #     print(early_stopping_score)
        #     break
        # else: early_stopping_score = correct_preds

    correct_preds = 0
    true_positive = 0
    true_negative = 0
    for i in range(n_batches):
        logits_batch = sess.run(tf.nn.sigmoid(logits), feed_dict={X: test_input[batch_size*i: batch_size*(i+1), :]})
        test_batch = test_target[batch_size*i:batch_size*(i+1), :]

        for j in range(len(test_batch)):
            if np.abs(logits_batch[j] - test_batch[j] < 0.01):
                correct_preds += 1
            if np.abs(test_batch[j] - logits_batch[j]) < 0.01:
                if test_batch[j] == 1:
                    true_positive += 1
                else: true_negative += 1

    print('correct predictions: {0}'.format(correct_preds))
    print('true positive: {0}'.format(true_positive/test_positive))
    print('true negative: {0}'.format(true_negative/test_negative))








