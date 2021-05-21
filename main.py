from data import DataSet
import random
import argparse
import time
import math
import tensorflow as tf
import numpy as np
import os
from sklearn.metrics import precision_recall_fscore_support
os.environ["CUDA_VISIBLE_DEVICES"]="0"

emb_dim=60
lr1 = 0.02 # For Margin-based loss...
lr2 = 0.01 # For Softmax...
# dataset args
parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', type=str, default='../GT_Th/adjectives/', help='dataset directory') # '../GT_Th/adjectives/
parser.add_argument('--learning_rate', type=float, default=lr1,help='initial learning rate')
parser.add_argument('--batch_size',type=int,default=500,help='mini batch size for SGD')
parser.add_argument('--num_epoch',type=int,default=4,help='number of epochs')
parser.add_argument('--margin',type=float,default=1.0,help='margin of a golden triplet and a corrupted one')
parser.add_argument('--embedding_dimension',type=int,default=emb_dim,help='dimension of words and relation embeddings')
parser.add_argument('--dissimilarity', type=str, default='sigmoid', help='using L1 or L2 distance as dissimilarity')
parser.add_argument('--num_input', default=300)
parser.add_argument('--num_hidden_1', default=80)
parser.add_argument('--num_hidden_2', default=emb_dim)


args = parser.parse_args()
print('args: {}'.format(args))
dataset = DataSet(data_dir=args.data_dir)

weights = {
    'encoder1_h1': tf.Variable(tf.random_normal([args.num_input, args.num_hidden_1])),
    'encoder1_h2': tf.Variable(tf.random_normal([args.num_hidden_1, args.num_hidden_2])),
    'encoder2_h1': tf.Variable(tf.random_normal([args.num_input, args.num_hidden_1])),
    'encoder2_h2': tf.Variable(tf.random_normal([args.num_hidden_1, args.num_hidden_2])),
}
biases = {
    'encoder1_b1': tf.Variable(tf.random_normal([args.num_hidden_1])),
    'encoder1_b2': tf.Variable(tf.random_normal([args.num_hidden_2])),
    'encoder2_b1': tf.Variable(tf.random_normal([args.num_hidden_1])),
    'encoder2_b2': tf.Variable(tf.random_normal([args.num_hidden_2])),
}

sources = tf.placeholder(dtype=tf.float32, shape=[None, args.num_input],name='source') #args.batch_size
targets = tf.placeholder(dtype=tf.float32, shape=[None, args.num_input], name='target')
GT = tf.placeholder(tf.float32, [None], name='GT')
dis_scores = tf.placeholder(dtype=tf.float32, shape=[None, 1], name='score')
Softmax_GT = tf.placeholder(tf.float32, [None,2], name='targets')

W = tf.Variable(tf.ones([6, 2]))#int(2*args.num_hidden_2 +5)

def encoder_OUT(x):
    # Encoder Hidden layer with sigmoid activation #1
    layer_1 = tf.nn.sigmoid(tf.add(tf.matmul(x, weights['encoder1_h1']),
                                   biases['encoder1_b1']))
    # Encoder Hidden layer with sigmoid activation #2
    layer_2 = tf.nn.sigmoid(tf.add(tf.matmul(layer_1, weights['encoder1_h2']),
                                   biases['encoder1_b2']))
    return layer_2

def encoder_IN(x):
    # Encoder Hidden layer with sigmoid activation #1
    layer_1 = tf.nn.sigmoid(tf.add(tf.matmul(x, weights['encoder2_h1']),
                                   biases['encoder2_b1']))
    # Encoder Hidden layer with sigmoid activation #2
    layer_2 = tf.nn.sigmoid(tf.add(tf.matmul(layer_1, weights['encoder2_h2']),
                                   biases['encoder2_b2']))
    return layer_2


def get_dissimilarity1(source,target):

    embedding_head = encoder_OUT(source)
    embedding_tail = encoder_IN (target)
    pro = tf.diag_part(tf.tensordot(embedding_head, embedding_tail, axes=[[1], [1]]))
    dissimilarity = tf.tanh(pro)
    return dissimilarity#tf.clip_by_value(dissimilarity, 1e-7, 1 - 1e-7)

def get_dissimilarity2(source,target):
    embedding_head = encoder_IN(source)
    embedding_tail = encoder_IN(target)

    pro = tf.diag_part(tf.tensordot(embedding_head, embedding_tail, axes=[[1], [1]]))
    dissimilarity = tf.tanh(pro)
    return dissimilarity#tf.clip_by_value(dissimilarity, 1e-7, 1 - 1e-7)

def get_dissimilarity3(source, target, score):

    _head1 = encoder_OUT(source)
    _head2 = encoder_OUT(target)

    _tail1 = encoder_IN (source)
    _tail2 = encoder_IN(target)

    #norm_h1 = tf.nn.l2_normalize(_head1, 0)
    #norm_h2 = tf.nn.l2_normalize(_head2, 0)
    #norm_t1 = tf.nn.l2_normalize(_tail1, 0)
    #norm_t2 = tf.nn.l2_normalize(_tail2, 0)

    syn_scores1 = tf.reshape(tf.reduce_sum(tf.multiply(_tail1,_tail2), 1), [4 * args.batch_size, 1])
    syn_scores2 = tf.reshape(tf.reduce_sum(tf.multiply(_head1, _head2), 1),[4 * args.batch_size, 1])
    ant_scores1 = tf.reshape(tf.reduce_sum(tf.multiply(_head1,_tail2), 1), [4 * args.batch_size, 1])
    ant_scores2 = tf.reshape(tf.reduce_sum(tf.multiply(_head2,_tail1), 1), [4 * args.batch_size, 1])
    result = tf.concat(axis=1, values=[syn_scores1, syn_scores2, ant_scores1, ant_scores2, score]) #tf.nn.sigmoid(_tail2-_tail1), tf.nn.sigmoid(_head2-_tail1)
    #result = tf.concat(axis=1, values=[_tail2-_tail1, _head1-_tail2])

    t2=np.ones([int(args.batch_size * 4),1])
    f = tf.concat([result,t2],axis=1)
    dissimilarity = tf.nn.softmax(tf.matmul(f,W))
    return tf.clip_by_value(dissimilarity, 1e-7, 1 - 1e-7)

def get_dissimilarity4(source, target, score):

    _head1 = encoder_OUT(source)
    _head2 = encoder_OUT(target)

    _tail1 = encoder_IN (source)
    _tail2 = encoder_IN(target)

    #norm_h1 = tf.nn.l2_normalize(_head1, 0)
    #norm_h2 = tf.nn.l2_normalize(_head2, 0)
    #norm_t1 = tf.nn.l2_normalize(_tail1, 0)
    #norm_t2 = tf.nn.l2_normalize(_tail2, 0)

    syn_scores1 = tf.reshape(tf.reduce_sum(tf.multiply(_tail1, _tail2), 1), [1986, 1])
    syn_scores2 = tf.reshape(tf.reduce_sum(tf.multiply(_head1, _head2), 1), [1986, 1])
    ant_scores1 = tf.reshape(tf.reduce_sum(tf.multiply(_head1, _tail2), 1), [1986, 1])
    ant_scores2 = tf.reshape(tf.reduce_sum(tf.multiply(_head2, _tail1), 1), [1986, 1])

    #ant_scores = [max(ant_scores1[i], ant_scores2[i]) for i, _ in enumerate(ant_scores1)]
    result = tf.concat(axis=1, values=[syn_scores1, syn_scores2, ant_scores1, ant_scores2, score]) #tf.nn.tanh(_tail2-_tail1), tf.nn.tanh(_head2-_tail1),
    t2=np.ones([1986,1])
    f= tf.concat([result,t2], axis=1)
    dissimilarity = tf.nn.softmax(tf.matmul(f, W))
    return tf.clip_by_value(dissimilarity, 1e-7, 1 - 1e-7)



def inference1(source,target):
    prediction = get_dissimilarity1(source,target)
    return prediction

def inference2(source,target):
    prediction = get_dissimilarity2(source,target)
    return prediction

def inference3(source,target,score):
    prediction = get_dissimilarity3(source,target,score)
    return prediction


def loss3(prediction, GT):
    #margin = tf.constant(value=1.0, shape=[4*args.batch_size])# Change bath_size accordingly...
    #pro = tf.multiply(prediction, GT)
    #loss = tf.reduce_sum(tf.nn.relu(margin - pro))
    #loss = tf.reduce_sum(-(GT * tf.log(prediction) + (margin - GT) * tf.log(margin - prediction)))
    loss = tf.reduce_mean(tf.reduce_sum((-GT * tf.log(prediction)) - ((1 - GT) * tf.log(1 - prediction)), reduction_indices=[1]))
    return loss


def loss(prediction, GT):
    margin = tf.constant(value=1.0, shape=[4*args.batch_size])# Change bath_size accordingly...
    pro = tf.multiply(prediction, GT)
    loss = tf.reduce_sum(tf.nn.relu(margin - pro))
    #loss = tf.reduce_sum(-(GT * tf.log(prediction) + (margin - GT) * tf.log(margin - prediction)))
    return loss


def train(loss):
    # add a scalar summary for the snapshot loss
    tf.summary.scalar(loss.op.name, loss)
    optimizer = tf.train.AdamOptimizer(learning_rate=args.learning_rate)
    train_op = optimizer.minimize(loss)
    return train_op

def train3(loss):
    # add a scalar summary for the snapshot loss
    tf.summary.scalar(loss.op.name, loss)
    optimizer = tf.train.AdamOptimizer(learning_rate=lr2)
    train_op = optimizer.minimize(loss)
    return train_op


prediction1 = inference1(sources, targets)
prediction2 = inference2(sources, targets)

prediction3 = inference3(sources, targets, dis_scores)
prediction4 = get_dissimilarity4(sources, targets, dis_scores)

loss1 = loss(prediction1, GT)
train_op1 = train(loss1)
loss2 = loss(prediction2, GT)
train_op2 = train(loss2)

loss_3 = loss3(prediction3, Softmax_GT)
train_op3 = train3(loss_3)

encoder_OUT_OP=encoder_OUT(sources)
encoder_IN_OP=encoder_IN(targets)

# initialize op
init_op = tf.global_variables_initializer()
saver=tf.train.Saver()



# open a session and run the training graph
session_config = tf.ConfigProto(log_device_placement=True)
session_config.gpu_options.allow_growth = True

with tf.Session(config=session_config) as sess:
    # run the initial operation
    print('initializing all variables...')
    sess.run(init_op)
    print('all variables initialized')

    ## Calculating number of batches to run...
    num_batch = len(dataset.ANT_train+ dataset.SYN_train) // args.batch_size

    # training
    print('start training...')
    start_total = time.time()
    for epoch in range(args.num_epoch):

        loss_epoch = 0
        start_train = time.time()
        for batch in range(num_batch):

            source, target, truth=dataset.next_batch_antonyms(args.batch_size)
            feed_dict_train1 = {sources:source, targets:target, GT:truth}
            _, loss_batch1=sess.run([train_op1, loss1], feed_dict=feed_dict_train1)

            source, target, truth=dataset.next_batch_synonyms(args.batch_size)
            feed_dict_train2={sources:source, targets:target, GT:truth}
            _,loss_batch2=sess.run([train_op2, loss2], feed_dict=feed_dict_train2)


            source, target, truth, score = dataset.next_train_batch(args.batch_size)
            feed_dict_train3 = {sources: source, targets: target, Softmax_GT: truth, dis_scores:score}
            _, _ = sess.run([train_op3, loss_3], feed_dict=feed_dict_train3)



            loss_batch=loss_batch1 + loss_batch2
            loss_epoch += loss_batch

            # print an overview every 10 batches
            if (batch + 1) % 1 == 0 or (batch + 1) == num_batch:
                print('epoch {}, batch {}, loss: {}'.format(epoch, batch, loss_batch))

        end_train = time.time()
        print('epoch {}, mean batch loss: {:.3f}, time elapsed last epoch: {:.3f}s'.format(epoch, loss_epoch / num_batch,
                                                                                            end_train - start_train))

    end_total = time.time()
    print('total time elapsed: {:.3f}s'.format(end_total - start_total))
    save_path = saver.save(sess, "./model/model.ckpt")
    print("Model saved in path: %s" % save_path)
    print('training finished')



    ## Write down Vocab Embeddings in file...
    sample = dataset.get_condensed_vectors()
    feed_dict_head={sources:sample}
    output_ = sess.run([encoder_OUT_OP],feed_dict=feed_dict_head)
    sorted_keys = sorted(dataset.vec_dic.keys())

    ## Write down head Embeddings.....
    with open('heads.emb', "w") as outfile:
        outfile.write(str(len(dataset.vec_dic.keys())) + ' ' + str(args.embedding_dimension) + '\n')
        for i in range(len(dataset.vec_dic.keys())):
            outfile.write(str(sorted_keys[i]) + ' ')
            op = output_[0][i]
            op = op.tolist()
            for j in range(len(op)):
                op[j] = round(op[j], 6)
            outfile.write(' '.join(map(str, op)))
            outfile.write('\n')

    ## Write down tail embeddings
    sample = dataset.get_condensed_vectors()
    feed_dict_tail = {targets: sample}
    output_ = sess.run([encoder_IN_OP], feed_dict=feed_dict_tail)
    sorted_keys = sorted(dataset.vec_dic.keys())
    with open('tails.emb', "w") as outfile:
        outfile.write(str(len(dataset.vec_dic.keys())) + ' ' + str(args.embedding_dimension) + '\n')
        for i in range(len(dataset.vec_dic.keys())):
            outfile.write(str(sorted_keys[i]) + ' ')
            op = output_[0][i]
            op = op.tolist()
            for j in range(len(op)):
                op[j] = round(op[j], 6)
            outfile.write(' '.join(map(str, op)))
            outfile.write('\n')
    
    ## Write down relation embeddings in a file...
    ## Write down output for the test cases...
    source, target, label, score = dataset.get_test_batch()
    y_pred = sess.run(prediction4, feed_dict={sources:source, targets:target, dis_scores:score})
    correct_prediction = tf.equal(tf.argmax(y_pred, 1), tf.argmax(label, 1))

    #predicted= y_pred.eval()
    #predicted = tf.argmax(y_pred, 1).eval()
    p_, r_, f1_, _ = precision_recall_fscore_support(tf.argmax(label, 1).eval(), tf.argmax(y_pred, 1).eval(), average='binary')
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    print("Accuracy: ", (sess.run(accuracy, feed_dict={sources: source, targets: target})))
    print("Average Prec={}, Recall={}, F1 ={}  ".format(round(p_, 3), round(r_, 3), round(f1_, 3)))
    #print(correct_prediction)
