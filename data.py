import random
import pandas as pd
from numpy.random import binomial
import gensim
import numpy as np
from os import path


class DataSet:
    def __init__(self, data_dir):
        self.data_dir = data_dir

        self.words_to_id = {}
        self.id_to_words = {}
        self.relation_to_id = {}
        self.id_to_relation = {}

        self.triplets_train_pool = set()  # {(id_head, id_relation, id_tail), ...}
        self.triplets_train = []  # [(id_head, id_relation, id_tail), ...]
        self.triplets_validate = []
        self.triplets_test = []
        self.ANT_train=[]
        self.SYN_train=[]

        self.num_words = 0
        self.num_relation = 0
        self.num_triplets_train = 0

        # for reducing false negative labels
        self.relation_dist = {}  # {relation, (head_per_tail, tail_per_head)}

        # load train, validate and test files
        self.load_data()

    def load_data(self):
        # read the words_to_id file
        print('loading words...')
        words_to_id_df = pd.read_csv(path.join(self.data_dir, 'train_vocab2id.csv'), sep=',')
        self.words_to_id = dict(zip(words_to_id_df['word'].astype(str), words_to_id_df['id']))
        self.id_to_words = dict(zip(words_to_id_df['id'], words_to_id_df['word'].astype(str)))
        self.num_words = len(self.words_to_id)
        print('got {} words'.format(self.num_words))

        # read the relation_to_id file
        print('loading relations...')
        relation_to_id_df = pd.read_csv(path.join(self.data_dir, 'train_rel2id.csv'), sep=',')
        self.relation_to_id = dict(zip(relation_to_id_df['relation'].astype(str), relation_to_id_df['id']))
        self.id_to_relation = dict(zip(relation_to_id_df['id'], relation_to_id_df['relation'].astype(str)))
        self.num_relation = len(self.relation_to_id)
        print('got {} relations'.format(self.num_relation))

        # read the train file
        print('loading train triplets...')
        triplets_train_df = pd.read_csv(path.join(self.data_dir, 'train_dev.csv'), sep=',')
        self.triplets_train = list(zip(
            [self.words_to_id[head] for head in triplets_train_df['head'].astype(str)],
            [self.relation_to_id[relation] for relation in triplets_train_df['relation'].astype(str)],
            [self.words_to_id[tail] for tail in triplets_train_df['tail'].astype(str)]))
        self.num_triplets_train = len(self.triplets_train)
        print('got {} triplets from training set'.format(self.num_triplets_train))
        # construct the train triplets pool
        newie=[]
        for (x,y,z) in self.triplets_train:
            newie.append((z,y,x))
        self.triplets_train=self.triplets_train+newie

        ## Analyzing Antonyms....
        ant_heads = [x for (x, y, z) in self.triplets_train if y == 1]
        ant_tails = [z for (x, y, z) in self.triplets_train if y == 1]
        ANT_dic = {}
        for idx in range(len(ant_heads)):
            head = ant_heads[idx]
            tail = ant_tails[idx]
            syn_list = [z for (x, y, z) in self.triplets_train if x == tail and y == 0]
            new = []
            for syn in syn_list:
                new = [z for (x, y, z) in self.triplets_train if x == syn and y == 0]
            syn_list = syn_list + new
            ant_list = [z for (x, y, z) in self.triplets_train if x == head and y == 1]
            new = []
            for ant in ant_list:
                new = [z for (x, y, z) in self.triplets_train if x == ant and y == 0]
            ant_list = ant_list + new

            ANT_dic[head] = list(set(syn_list + ant_list))

        ## Analyzing Synonyms...
        syn_heads = [x for (x, y, z) in self.triplets_train if y == 0]
        syn_tails = [z for (x, y, z) in self.triplets_train if y == 0]
        SYN_dic = {}
        for idx in range(len(ant_heads)):
            head = syn_heads[idx]
            tail = syn_tails[idx]
            list1_ = [z for (x, y, z) in self.triplets_train if x == tail and y == 0]
            new = []
            for syn in list1_:
                new = [z for (x, y, z) in self.triplets_train if x == syn and y == 0]
            list1_ = list1_ + new
            list2_ = [z for (x, y, z) in self.triplets_train if x == head and y == 0]
            new = []
            for syn in list2_:
                new = [z for (x, y, z) in self.triplets_train if x == syn and y == 0]
            list2_ = list2_ + new
            SYN_dic[head] = list(set(list1_ + list2_))

        NoN_triplets = []
        for k, v in ANT_dic.items():
            ant = [(k, 1, x) for x in v]
            NoN_triplets = NoN_triplets + ant

        for k, v in SYN_dic.items():
            syn = [(k, 0, x) for x in v]
            NoN_triplets = NoN_triplets + syn

        # Load Embedding Dictionary...
        model = gensim.models.KeyedVectors.load_word2vec_format('../filtered_embeddings_glove', binary=False)
        self.model = model
        self.vec_dic ={}
        for key in model.vocab.keys():
            self.vec_dic[key]= model.word_vec(key)

        self.triplets_train=self.triplets_train + NoN_triplets
        ant_train = []
        for (x, y, z) in self.triplets_train:
            if y == 1:
                if(self.id_to_words[x] in self.vec_dic.keys() and self.id_to_words[z] in self.vec_dic.keys()):
                    ant_train.append((z, y, x))
                    ant_train.append((x,y,z))

        syn_train = []
        for (x, y, z) in self.triplets_train:
            if y == 0:
                if (self.id_to_words[x] in self.vec_dic.keys() and self.id_to_words[z] in self.vec_dic.keys()):
                    syn_train.append((z, y, x))
                    syn_train.append((x,y,z))
        self.ANT_train = ant_train
        self.SYN_train = syn_train
        self.triplets_train_pool = set(self.triplets_train + NoN_triplets)

    def next_batch_antonyms(self, batch_size):
        batch_positive1 = random.sample(self.ANT_train, batch_size)
        batch_positive2 = random.sample(self.ANT_train, batch_size)

        batch_negative1 = random.sample(self.SYN_train, batch_size)
        batch_negative2 = random.sample(self.SYN_train, batch_size)

        '''
        # construct negative batch
        batch_negative2=[]
        for id_head, id_relation, id_tail in batch_positive1:
            while True:
                id_tail_corrupted = self.words_to_id[random.sample(list(self.words_to_id.keys()), 1)[0]]
                if (id_head, id_relation, id_tail_corrupted) not in self.triplets_train_pool:
                    break
            batch_negative2.append((id_head, id_relation, id_tail_corrupted))
        '''
        batch_positive = batch_positive1+batch_positive2
        batch_negative = batch_negative1+batch_negative2
        batch = batch_positive + batch_negative

        source = [self.vec_dic[self.id_to_words[x]] for (x,_,z) in batch]
        target = [self.vec_dic[self.id_to_words[z]] for (x,_,z) in batch]

        #source=self.vec_dic[self.id_to_words[batch[0][0]]]
        #target = self.vec_dic[self.id_to_words[batch[0][2]]]
        #for (x,_,z) in batch[1:]:
        #    sou = self.vec_dic[self.id_to_words[x]]
        #    tar = self.vec_dic[self.id_to_words[z]]
        #    source=np.vstack((source, sou))
        #    target=np.vstack((target,tar))

        truth = [1] * 2 * batch_size + [-1] * 2 * batch_size
        #random.Random(42).shuffle(batch)
        #random.Random(42).shuffle(truth)
        return source, target, truth

    def next_batch_synonyms(self, batch_size,):
        batch_positive1 = random.sample(self.SYN_train, batch_size)
        batch_positive2 = random.sample(self.SYN_train, batch_size)

        batch_negative1 = random.sample(self.ANT_train, batch_size)
        batch_negative2 = random.sample(self.ANT_train, batch_size)
        #batch_negative3 = random.sample(self.ANT_train, batch_size)

        '''
        # construct negative batch
        batch_negative2 = []
        for id_head, id_relation, id_tail in batch_positive1:
            r = random.random()
            if (r> 0.5):
                while True:
                    id_tail_corrupted = self.words_to_id[random.sample(list(self.words_to_id.keys()), 1)[0]]
                    if (id_head, id_relation, id_tail_corrupted) not in self.triplets_train_pool:
                        break
                batch_negative2.append((id_head, id_relation, id_tail_corrupted))
            else:
                while True:
                    id_head_corrupted = self.words_to_id[random.sample(list(self.words_to_id.keys()), 1)[0]]
                    if (id_head_corrupted, id_relation, id_tail) not in self.triplets_train_pool:
                        break
                batch_negative2.append((id_head_corrupted, id_relation, id_tail))
        '''
        batch_positive = batch_positive1 + batch_positive2
        batch_negative = batch_negative1 + batch_negative2 #+ batch_negative3

        batch = batch_positive + batch_negative
        #source=self.vec_dic[self.id_to_words[batch[0][0]]]
        #target = self.vec_dic[self.id_to_words[batch[0][2]]]
        #for (x,_,z) in batch[1:]:
        #    sou = self.vec_dic[self.id_to_words[x]]
        #    tar = self.vec_dic[self.id_to_words[z]]
        #    source=np.vstack((source, sou))
        #    target=np.vstack((target,tar))

        source = [self.vec_dic[self.id_to_words[x]] for (x,_,z) in batch]
        target = [self.vec_dic[self.id_to_words[z]] for (x,_,z) in batch]
        #batch = batch_positive + batch_negative
        truth = [1] * 2 * batch_size + [-1] * 2 * batch_size
        #random.Random(42).shuffle(batch)
        #random.Random(42).shuffle(targets)
        return source, target, truth


    def next_train_batch(self, batch_size):
        batch_positive1 = random.sample(self.ANT_train, batch_size)
        batch_positive2 = random.sample(self.ANT_train, batch_size)
        batch_negative1 = random.sample(self.SYN_train, batch_size)
        batch_negative2 = random.sample(self.SYN_train, batch_size)

        # construct negative batch
        '''
        batch_negative2=[]
        for id_head, id_relation, id_tail in batch_positive1:
            r = random.random()
            if r >0.5:
                while True:
                    id_tail_corrupted = self.words_to_id[random.sample(list(self.words_to_id.keys()), 1)[0]]
                    if (id_head, id_relation, id_tail_corrupted) not in self.triplets_train_pool:
                        break
                batch_negative2.append((id_head, id_relation, id_tail_corrupted))
            else:
                while True:
                    id_head_corrupted = self.words_to_id[random.sample(list(self.words_to_id.keys()), 1)[0]]
                    if (id_head_corrupted, id_relation, id_tail) not in self.triplets_train_pool:
                        break
                batch_negative2.append((id_head_corrupted, id_relation, id_tail))
        '''
        batch_positive = batch_positive1+batch_positive2
        batch_negative = batch_negative1+batch_negative2
        batch = batch_positive + batch_negative

        score = [self.model.similarity(self.id_to_words[x], self.id_to_words[z]) for (x, _, z) in batch]
        source = [self.vec_dic[self.id_to_words[x]] for (x,_,z) in batch]
        target = [self.vec_dic[self.id_to_words[z]] for (x,_,z) in batch]

        score = np.asarray(score)
        score = score.reshape(4*batch_size,1)

        #truth = [1] * 2 * batch_size + [-1] * 2 * batch_size
        truth = [[0,1]] * 2 * batch_size + [[1, 0]] * 2 * batch_size
        return source, target, truth, score

    def get_condensed_vectors(self):
        keys = sorted(self.vec_dic.keys(), reverse=False)
        result = []
        for key in keys:
            result.append(self.vec_dic[key])
        result = np.asarray(result)
        return result

    def get_test_batch(self):
        # read the words_to_id for Testing
        print('loading Test triplets...')
        triplets_test_df = pd.read_csv(path.join(self.data_dir, 'all_test.csv'), sep=',')
        self.triplets_test = list(zip(
            [self.words_to_id[head] for head in triplets_test_df['head'].astype(str)],
            [self.relation_to_id[relation] for relation in triplets_test_df['relation'].astype(str)],
            [self.words_to_id[tail] for tail in triplets_test_df['tail'].astype(str)]))
        self.num_triplets_test = len(self.triplets_test)
        print('got {} triplets from Testing set'.format(len(self.triplets_test)))

        score = [self.model.similarity(self.id_to_words[x], self.id_to_words[z]) for (x, _, z) in self.triplets_test]
        source = [self.vec_dic[self.id_to_words[x]] for (x,_,z) in self.triplets_test]
        target = [self.vec_dic[self.id_to_words[z]] for (x,_,z) in self.triplets_test]
        lls = [y for (x,y,z) in self.triplets_test]
        score = np.asarray(score)
        score = score.reshape(1986,1)

        label=[]
        GT=[]

        for entry in lls:
            if entry == 1:
                label.append([0,1])
                GT.append(1)
            else:
                label.append([1,0])
                GT.append(0)
        return source, target, label, score#, GT
