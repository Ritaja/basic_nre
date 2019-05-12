import tensorflow as tf
import numpy as np
import os
import data_loader
import embedding

dataset_dir = "/home/piyantatai/MachineLearning/repos/basic_nlp/data"

# The first 3 parameters are train / test data file name, word embedding file name and relation-id mapping file name respectively.
train_loader = data_loader.JSONLoader(dataset_dir, os.path.join(dataset_dir, 'train.json'),
                                      os.path.join(
                                          dataset_dir, 'word_vec.json'),
                                      os.path.join(dataset_dir, 'rel2id.json'),
                                      mode=data_loader.JSONLoader.MODE_RELFACT_BAG,
                                      shuffle=True)
test_loader = data_loader.JSONLoader(dataset_dir, os.path.join(dataset_dir, 'test.json'),
                                     os.path.join(
                                         dataset_dir, 'word_vec.json'),
                                     os.path.join(dataset_dir, 'rel2id.json'),
                                     mode=data_loader.JSONLoader.MODE_ENTPAIR_BAG,
                                     shuffle=False)

sess = None  # default graph session

# ====== Embedding ======
word_embedding_train = embedding.word_position_embedding(
    train_loader.data_word, train_loader.word_vec_mat, train_loader.data_pos1, train_loader.data_pos2)

word_embedding_test = embedding.word_position_embedding(
    test_loader.data_word, test_loader.word_vec_mat, test_loader.data_pos1, test_loader.data_pos2)

# CNN cell


def __cnn_cell__(x, hidden_size=230, kernel_size=3, stride_size=1):
    x = tf.layers.conv1d(inputs=x,
                         filters=hidden_size,
                         kernel_size=kernel_size,
                         strides=stride_size,
                         padding='same',
                         kernel_initializer=tf.contrib.layers.xavier_initializer())
    return x

# Picewise pooling

def __piecewise_pooling__(x, mask):
    mask_embedding = tf.constant(
        [[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=np.float32)
    mask = tf.nn.embedding_lookup(mask_embedding, mask)
    hidden_size = x.shape[-1]
    x = tf.reduce_max(tf.expand_dims(mask * 100, 2) +
                      tf.expand_dims(x, 3), axis=1) - 100
    return tf.reshape(x, [-1, hidden_size * 3])

# dropout from contribution stack


def __dropout__(x, keep_prob=1.0):
    return tf.contrib.layers.dropout(x, keep_prob=keep_prob)

# PCNN function


def pcnn(x, mask, hidden_size=230, kernel_size=3, stride_size=1, activation=tf.nn.relu, var_scope=None, keep_prob=1.0):
    with tf.variable_scope(var_scope or "pcnn", reuse=tf.AUTO_REUSE):
        max_length = x.shape[1]
        x = __cnn_cell__(x, hidden_size, kernel_size, stride_size)
        x = __piecewise_pooling__(x, mask)
        x = activation(x)
        x = __dropout__(x, keep_prob)
        return x


# ====== Encoder ======
x_train = pcnn(word_embedding_train, train_loader.data_length, keep_prob=0.5)
x_test = pcnn(word_embedding_test, test_loader.data_mask, keep_prob=1.0)

# attention

def __logit__(x, rel_tot, var_scope=None):
    with tf.variable_scope(var_scope or 'logit', reuse=tf.AUTO_REUSE):
        relation_matrix = tf.get_variable('relation_matrix', shape=[rel_tot, x.shape[1]], dtype=tf.float32, initializer=tf.contrib.layers.xavier_initializer())
        bias = tf.get_variable('bias', shape=[rel_tot], dtype=tf.float32, initializer=tf.contrib.layers.xavier_initializer())
        logit = tf.matmul(x, tf.transpose(relation_matrix)) + bias
    return logit

def __attention_train_logit__(x, query, rel_tot, var_scope=None):
    with tf.variable_scope(var_scope or 'logit', reuse=tf.AUTO_REUSE):
        relation_matrix = tf.get_variable('relation_matrix', shape=[rel_tot, x.shape[1]], dtype=tf.float32, initializer=tf.contrib.layers.xavier_initializer())
        bias = tf.get_variable('bias', shape=[rel_tot], dtype=tf.float32, initializer=tf.contrib.layers.xavier_initializer())
    current_relation = tf.nn.embedding_lookup(relation_matrix, query)
    attention_logit = tf.reduce_sum(current_relation * x, -1) # sum[(n', hidden_size) \dot (n', hidden_size)] = (n)
    return attention_logit

def __attention_test_logit__(x, rel_tot, var_scope=None):
    with tf.variable_scope(var_scope or 'logit', reuse=tf.AUTO_REUSE):
        relation_matrix = tf.get_variable('relation_matrix', shape=[rel_tot, x.shape[1]], dtype=tf.float32, initializer=tf.contrib.layers.xavier_initializer())
        bias = tf.get_variable('bias', shape=[rel_tot], dtype=tf.float32, initializer=tf.contrib.layers.xavier_initializer())
    attention_logit = tf.matmul(x, tf.transpose(relation_matrix)) # (n', hidden_size) x (hidden_size, rel_tot) = (n', rel_tot)
    return attention_logit


# TODO: need to understand
def bag_attention(x, scope, query, rel_tot, is_training, var_scope=None, dropout_before=False, keep_prob=1.0):
    with tf.variable_scope(var_scope or "attention", reuse=tf.AUTO_REUSE):
        if is_training:  # training
            if dropout_before:
                x = __dropout__(x, keep_prob)
            bag_repre = []
            attention_logit = __attention_train_logit__(x, query, rel_tot)
            for i in range(scope.shape[0]):
                bag_hidden_mat = x[scope[i][0]:scope[i][1]]
                attention_score = tf.nn.softmax(
                    attention_logit[scope[i][0]:scope[i][1]], -1)
                # (1, n') x (n', hidden_size) = (1, hidden_size) -> (hidden_size)
                bag_repre.append(tf.squeeze(
                    tf.matmul(tf.expand_dims(attention_score, 0), bag_hidden_mat)))
            bag_repre = tf.stack(bag_repre)
            if not dropout_before:
                bag_repre = __dropout__(bag_repre, keep_prob)
            return __logit__(bag_repre, rel_tot), bag_repre
        else:  # testing
            attention_logit = __attention_test_logit__(
                x, rel_tot)  # (n, rel_tot)
            bag_repre = []
            bag_logit = []
            for i in range(scope.shape[0]):
                bag_hidden_mat = x[scope[i][0]:scope[i][1]]
                attention_score = tf.nn.softmax(tf.transpose(
                    attention_logit[scope[i][0]:scope[i][1], :]), -1)  # softmax of (rel_tot, n')
                # (rel_tot, n') \dot (n', hidden_size) = (rel_tot, hidden_size)
                bag_repre_for_each_rel = tf.matmul(
                    attention_score, bag_hidden_mat)
                bag_logit_for_each_rel = __logit__(
                    bag_repre_for_each_rel, rel_tot)  # -> (rel_tot, rel_tot)
                bag_repre.append(bag_repre_for_each_rel)
                # could be improved by sigmoid?
                bag_logit.append(tf.diag_part(
                    tf.nn.softmax(bag_logit_for_each_rel, -1)))
            bag_repre = tf.stack(bag_repre)
            bag_logit = tf.stack(bag_logit)
            return bag_logit, bag_repre

# Selector
_train_logit, train_repre = bag_attention(x_train, train_loader.scope, train_loader.ins_label, train_loader.rel_tot, True, keep_prob=0.5)
_test_logit, test_repre = bag_attention(x_test, test_loader.scope, self.ins_label, test_loader.rel_tot, False, keep_prob=1.0)

# softmax cross entropy

def softmax_cross_entropy(x, label, rel_tot, weights_table=None, weights=1.0, var_scope=None):
    with tf.variable_scope(var_scope or "loss", reuse=tf.AUTO_REUSE):
        if weights_table is not None:
            weights = tf.nn.embedding_lookup(weights_table, label)
        label_onehot = tf.one_hot(indices=label, depth=rel_tot, dtype=tf.int32)
        loss = tf.losses.softmax_cross_entropy(onehot_labels=label_onehot, logits=x, weights=weights)
        tf.summary.scalar('loss', loss)
        return loss

def get_weights():
        with tf.variable_scope("weights_table", reuse=tf.AUTO_REUSE):
            print("Calculating weights_table...")
            _weights_table = np.zeros((train_loader.rel_tot), dtype=np.float32)
            for i in range(len(train_loader.data_rel)):
                _weights_table[train_loader.data_rel[i]] += 1.0 
            _weights_table = 1 / (_weights_table ** 0.05)
            weights_table = tf.get_variable(name='weights_table', dtype=tf.float32, trainable=False, initializer=_weights_table)
            print("Finish calculating")
        return weights_table

# Classifier
loss = softmax_cross_entropy(_train_logit, train_loader.label, train_loader.rel_tot, weights_table=get_weights())
