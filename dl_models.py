import pandas as pd
import numpy as np
import itertools
import datetime
import pickle, random

from collections import Counter
from sklearn.model_selection import train_test_split
from time import time
import pickle

from keras.models import load_model, Model, Sequential
from keras.layers import Bidirectional, Input, Embedding, Activation, Dense, Concatenate, Reshape, Dropout
from keras.layers.recurrent import LSTM
import keras.backend as K
from keras.callbacks import Callback,ModelCheckpoint, EarlyStopping
from keras.layers import Flatten, Lambda
from keras.layers.convolutional import Conv1D
from keras.layers.pooling import MaxPooling1D
from sklearn.metrics import classification_report, confusion_matrix

import keras.backend as K
import tensorflow as tf

from keras import initializers
from keras.callbacks import CSVLogger
import keras
# import nltk, re
# nltk.download('stopwords')
# nltk.download('punkt')
# nltk.download('wordnet')
# from nltk.stem import WordNetLemmatizer 
# from nltk.corpus import stopwords
# from nltk.tokenize import word_tokenize

from sentence_transformers import SentenceTransformer
bert_model = SentenceTransformer('bert-base-uncased')

# lemmatizer = WordNetLemmatizer()
# stop_words = set(stopwords.words('english'))
# r_anum = re.compile(r'([^\sa-z.(?)!])+')
# r_white = re.compile(r'[\s.(?)!]+')

# def preprocess(post):
    
#     post = word_tokenize(post)
#     post_preproc = [lemmatizer.lemmatize(w) for w in post if not w.strip().lower() in stop_words]
#     post = str(' '.join(post)).lower()
#     row_clean = r_white.sub(' ', r_anum.sub('', post.lower())).strip()
#     return row_clean





def load_data_and_labels(data):

    text = data['text'].values
    text = [str(temp).split(' ') for temp in text]

    return text, data['label'].values

def pad_sentences(sentences, max_len, padding_word="<PAD/>"):
    
    # max_len = max(len(x) for x in sentences)
    # min_len =  min(len(x) for x in sentences)
    # max_len += 1

    # print(max_len, min_len)
    padded_sentences = list()
    for a in sentences:
        if len(a) > max_len:
            a = a[:max_len]
        else:
            a += [padding_word] * (max_len - len(a))
        padded_sentences.append(a)
    
    return padded_sentences

def build_vocab(sentences):
    
    word_counts = Counter(itertools.chain(*sentences))
    X_plot = list()
    Y_plot = list()
    till = 0 
    for a,b in word_counts.most_common():
      # if b<=5:
      #   break
        till += 1
      # print(a,b)
    print(len(word_counts.most_common(till)))
    vocabulary_inv = [x[0] for x in word_counts.most_common(till)]
    vocabulary_inv.append('<UNK>')
    vocabulary = {x: i for i, x in enumerate(vocabulary_inv)}
    return [vocabulary, vocabulary_inv]

def build_input_data(sentences, labels, vocabulary):
    
    x = []
    for sentence in sentences:
        temp = []
        for word in sentence:
            if str(word) in vocabulary:
                temp.append(vocabulary[str(word)])
            else:
                temp.append(vocabulary['<UNK>'])
        x.append(temp)

    y = np.array(labels)
    return [np.array(x), y]


def load_data(max_len, data):

    sentences, labels = load_data_and_labels(data)

    sentences = pad_sentences(sentences, max_len)
    print(len(sentences), len(sentences[0]))

    vocabulary, vocabulary_inv = build_vocab(sentences)

    X, y = build_input_data(sentences, labels, vocabulary)
    return [X, y, vocabulary, vocabulary_inv]



def mcc(y_true, y_pred):
    y_pred_pos = K.round(K.clip(y_pred, 0, 1))
    y_pred_neg = 1 - y_pred_pos

    y_pos = K.round(K.clip(y_true, 0, 1))
    y_neg = 1 - y_pos

    tp = K.sum(y_pos * y_pred_pos)
    tn = K.sum(y_neg * y_pred_neg)

    fp = K.sum(y_neg * y_pred_pos)
    fn = K.sum(y_pos * y_pred_neg)

    numerator = (tp * tn - fp * fn)
    denominator = K.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))

    return numerator / (denominator + K.epsilon())

def confusion(y_true, y_pred):
    y_pred_pos = K.round(K.clip(y_pred, 0, 1))
    y_pred_neg = 1 - y_pred_pos
    y_pos = K.round(K.clip(y_true, 0, 1))
    y_neg = 1 - y_pos
    tp = K.sum(y_pos * y_pred_pos) / (K.sum(y_pos) + + K.epsilon())
    tn = K.sum(y_neg * y_pred_neg) / (K.sum(y_neg) + + K.epsilon())
    return {'true_pos': tp, 'true_neg': tn}

def get_f1(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))

    precision = true_positives / (predicted_positives + K.epsilon())
    recall = true_positives / (possible_positives + K.epsilon())
    f1_val = 2*(precision*recall)/(precision+recall+K.epsilon())
    return f1_val

def macro_f1(y_true, y_pred):
    y_pred_pos = K.round(K.clip(y_pred, 0, 1))
    y_pred_neg = 1 - y_pred_pos

    y_pos = K.round(K.clip(y_true, 0, 1))
    y_neg = 1 - y_pos

    tp = K.sum(y_pos * y_pred_pos)
    tn = K.sum(y_neg * y_pred_neg)

    fp = K.sum(y_neg * y_pred_pos)
    fn = K.sum(y_pos * y_pred_neg)
    p = tp / (tp + fp + K.epsilon())
    r = tp / (tp + fn + K.epsilon())

    f1 = 2*p*r / (p+r+K.epsilon())
    f1 = tf.where(tf.math.is_nan(f1), tf.zeros_like(f1), f1)
    return K.mean(f1)
    # return (f1_p + f1_n)/2.0

def recall(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall

def precision(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision

def f1_loss(y_true, y_pred):
    
    tp = K.sum(K.cast(y_true*y_pred, 'float'), axis=0)
    tn = K.sum(K.cast((1-y_true)*(1-y_pred), 'float'), axis=0)
    fp = K.sum(K.cast((1-y_true)*y_pred, 'float'), axis=0)
    fn = K.sum(K.cast(y_true*(1-y_pred), 'float'), axis=0)

    p = tp / (tp + fp + K.epsilon())
    r = tp / (tp + fn + K.epsilon())

    f1 = 2*p*r / (p+r+K.epsilon())
    f1 = tf.where(tf.math.is_nan(f1), tf.zeros_like(f1), f1)
    return 1 - K.mean(f1)


from keras.layers import TimeDistributed, Embedding, Dense, Input, Flatten, Conv1D, GlobalMaxPooling1D, Dropout, LSTM, GRU, Bidirectional, concatenate

def tunable_embed_apply(word_cnt_post, vocab_size, embed_mat, word_feat_name):
    input_seq = Input(shape=(word_cnt_post,), name=word_feat_name+'_t')
    embed_layer = Embedding(vocab_size, embed_mat.shape[1], embeddings_initializer=initializers.Constant(embed_mat), input_length=word_cnt_post, name=word_feat_name)
    embed_layer.trainable = True
    embed_l = embed_layer(input_seq)
    return input_seq, embed_l

def rnn_dense_apply(rnn_seq, input_seq, rnn_dim, att_dim, dropO2, nonlin, out_vec_size, rnn_type):
    if rnn_type == 'lstm':
        blstm_l = Bidirectional(LSTM(rnn_dim, return_sequences=(att_dim > 0)))(rnn_seq)
    else:
        blstm_l = Bidirectional(GRU(rnn_dim, return_sequences=(att_dim > 0)))(rnn_seq)
    if att_dim > 0:
        blstm_l, att_w = attLayer_hier(att_dim)(blstm_l)
    return apply_dense(input_seq, dropO2, blstm_l, nonlin, out_vec_size)

def apply_dense(input_seq, dropO2, post_vec, nonlin, out_vec_size):
    dr2_l = Dropout(dropO2)(post_vec)
    out_vec = Dense(out_vec_size, activation=nonlin)(dr2_l)
    return Model(input_seq, out_vec)

def c_bilstm(word_cnt_post, word_f, rnn_dim, att_dim, dropO1, dropO2, nonlin, out_vec_size, rnn_type, num_cnn_filters, kernel_sizes):
    if 'embed_mat' in word_f:
        input_seq, embedded_seq = tunable_embed_apply(word_cnt_post, len(word_f['embed_mat']), word_f['embed_mat'], 'random_embeds')
        dr1_l = Dropout(dropO1)(embedded_seq)
    else:
        input_seq = Input(shape=(word_cnt_post, word_f['dim_shape'][-1]))
        dr1_l = Dropout(dropO1)(input_seq)

    conv_l_list = []
    for k in kernel_sizes:
        conv_t = Conv1D(num_cnn_filters, k, padding='same', activation='relu')(dr1_l)
        conv_l_list.append(conv_t)
    conc_mat = concatenate(conv_l_list)
    return rnn_dense_apply(conc_mat, input_seq, rnn_dim, att_dim, dropO2, nonlin, out_vec_size, rnn_type), None


from keras.engine.topology import Layer

class attLayer_hier(Layer):
    def __init__(self, attention_dim, **kwargs):
        self.init = initializers.get('glorot_uniform')
        self.attention_dim = attention_dim
        super(attLayer_hier, self).__init__(**kwargs)

    def build(self, input_shape):
        self.W = self.add_weight(name = 'W', shape = (input_shape[-1], self.attention_dim), initializer=self.init, trainable=True)
        self.b = self.add_weight(name = 'b', shape = (self.attention_dim, ), initializer=self.init, trainable=True)
        self.u = self.add_weight(name = 'u', shape = (self.attention_dim, 1), initializer=self.init, trainable=True)
        super(attLayer_hier, self).build(input_shape)

    def call(self, x, mask=None):
        uit = K.tanh(K.bias_add(K.dot(x, self.W), self.b))
        ait = K.dot(uit, self.u)
        ait = K.squeeze(ait, -1)
        ait = K.exp(ait)

        if mask is not None:
            ait *= K.cast(mask, K.floatx())

        ait /= K.cast(K.sum(ait, axis=1, keepdims=True) + K.epsilon(), K.floatx())
        exp_ait = K.expand_dims(ait)
        weighted_input = x * exp_ait
        output = K.sum(weighted_input, axis=1)

        return [output, ait]

    def compute_output_shape(self, input_shape):
        return [(input_shape[0], input_shape[-1]), (input_shape[0], input_shape[1])]

    def get_config(self):
        config = {'attention_dim': self.attention_dim}
        base_config = super(attLayer_hier, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

def rnn_sen_embed(word_cnt_sent, word_emb_len, dropO1, rnn_dim, att_dim, rnn_type):
    w_emb_input_seq = Input(shape=(word_cnt_sent, word_emb_len), name='emb_input')
    if rnn_type == 'lstm':
        blstm_l = Bidirectional(LSTM(rnn_dim, return_sequences=(att_dim > 0)))(w_emb_input_seq)
    else:
        blstm_l = Bidirectional(GRU(rnn_dim, return_sequences=(att_dim > 0)))(w_emb_input_seq)
    if att_dim > 0:
        blstm_l, att_w = attLayer_hier(att_dim)(blstm_l)
        return Model(w_emb_input_seq, blstm_l), Model(w_emb_input_seq, att_w)
    else:
        return Model(w_emb_input_seq, blstm_l)

def cnn_sen_embed(word_cnt_sent, word_emb_len, dropO1, num_cnn_filters, max_pool_k_val, kernel_sizes):
    w_emb_input_seq = Input(shape=(word_cnt_sent, word_emb_len), name='emb_input')
    conv_l_list = []
    for k in kernel_sizes:
        conv_t = Conv1D(num_cnn_filters, k, padding='same', activation='relu')(w_emb_input_seq)
        if max_pool_k_val == 1:
            pool_t = GlobalMaxPooling1D()(conv_t)
        else:
            pool_t = kmax_pooling(max_pool_k_val)(conv_t)
        conv_l_list.append(pool_t)
    feat_vec = concatenate(conv_l_list)
    return Model(w_emb_input_seq, feat_vec)

def flat_embed(enc_algo, word_emb_seq, word_cnt_post, word_emb_len, dropO1, num_cnn_filters, max_pool_k_val, kernel_sizes, rnn_dim, att_dim, rnn_type, att_outputs):
    if enc_algo == "rnn":
        if att_dim > 0:
            rnn_mod, att_mod = rnn_sen_embed(word_cnt_post, word_emb_len, dropO1, rnn_dim, att_dim, rnn_type)
            att_outputs.append(att_mod(word_emb_seq))
        else:
            rnn_mod = rnn_sen_embed(word_cnt_post, word_emb_len, dropO1, rnn_dim, att_dim, rnn_type)
        return rnn_mod(word_emb_seq), att_outputs
    elif enc_algo == "cnn":
        cnn_mod = cnn_sen_embed(word_cnt_post, word_emb_len, dropO1, num_cnn_filters, max_pool_k_val, kernel_sizes)
        return cnn_mod(word_emb_seq), att_outputs
    elif enc_algo == "comb_cnn_rnn":
        if att_dim > 0:
            rnn_mod, att_mod = rnn_sen_embed(word_cnt_post, word_emb_len, dropO1, rnn_dim, att_dim, rnn_type)
            rnn_emb_output = rnn_mod(word_emb_seq)
            att_outputs.append(att_mod(word_emb_seq))
        else:
            rnn_emb_output = rnn_sen_embed(word_cnt_post, word_emb_len, dropO1, rnn_dim, att_dim, rnn_type)(word_emb_seq)
        cnn_emb_output = cnn_sen_embed(word_cnt_post, word_emb_len, dropO1, num_cnn_filters, max_pool_k_val, kernel_sizes)(word_emb_seq)

        return concatenate([cnn_emb_output, rnn_emb_output]), att_outputs

def add_word_emb_p_flat(model_inputs, word_emb_input, word_f_word_emb, word_f_emb_size, enc_algo, m_id, p_dict):
        model_inputs.append(word_emb_input)
        if m_id in p_dict:
            p_dict[m_id]["comb_feature_list"].append(word_f_word_emb)
            p_dict[m_id]["word_emb_len"] += word_f_emb_size
            p_dict[m_id]["enc_algo"] = enc_algo
        else:
            p_dict[m_id] = {}
            p_dict[m_id]["comb_feature_list"] = [word_f_word_emb]
            p_dict[m_id]["word_emb_len"] = word_f_emb_size 
            p_dict[m_id]["enc_algo"] = enc_algo


def flat_fuse(word_cnt_post, rnn_dim, att_dim, word_feats, sen_enc_feats, dropO1, dropO2, nonlin, out_vec_size, rnn_type, stack_rnn_flag, num_cnn_filters, max_pool_k_val, kernel_sizes):
    p_dict = {}
    model_inputs = []
    att_outputs = []

    for word_feat in word_feats:
        if 'embed_mat' in word_feat:
            word_f_input, word_f_word_emb_raw = tunable_embed_apply(word_cnt_post, len(word_feat['embed_mat']), word_feat['embed_mat'], "name")
            word_f_word_emb = Dropout(dropO1)(word_f_word_emb_raw)
            add_word_emb_p_flat(model_inputs, word_f_input, word_f_word_emb, word_feat['embed_mat'].shape[-1], word_feat['s_enc'], word_feat['m_id'], p_dict)
        else:
            word_f_input = Input(shape=(word_cnt_post, word_feat['dim_shape'][-1]), name=word_feat['emb'])
            word_f_word_emb = Dropout(dropO1)(word_f_input)
            add_word_emb_p_flat(model_inputs, word_f_input, word_f_word_emb, word_feat['dim_shape'][-1], word_feat['s_enc'], word_feat['m_id'], p_dict)

    post_vec_list = []    
    for my_dict in p_dict.values():
        my_dict["word_emb"] = concatenate(my_dict["comb_feature_list"]) if len(my_dict["comb_feature_list"]) > 1 else my_dict["comb_feature_list"][0]
        flat_emb, att_outputs = flat_embed(my_dict["enc_algo"], my_dict["word_emb"], word_cnt_post, my_dict["word_emb_len"], dropO1, num_cnn_filters, max_pool_k_val, kernel_sizes, rnn_dim, att_dim, rnn_type, att_outputs)
        post_vec_list.append(flat_emb)

    post_vec = concatenate(post_vec_list) if len(post_vec_list) > 1 else post_vec_list[0]
    if len(model_inputs) == 1:
        model_inputs = model_inputs[0]
    att_mod = Model(model_inputs, att_outputs) if att_outputs else None
    return apply_dense(model_inputs, dropO2, post_vec, nonlin, out_vec_size), att_mod

from keras import initializers

def init(seed, length):
    np.random.seed(seed)
    tf.compat.v1.set_random_seed(seed)
    random.seed(seed)
    max_len = length

    train_data = pd.read_csv('data_2k.csv', encoding = 'unicode_escape')

    y = train_data['label'].values
    train_data = train_data[['text', 'label']]
    train_data['text'] = train_data.apply(lambda row: np.str_(row[0]), axis=1)

    # print("Train data head: ", train_data.head())

    data = train_data
    # data['text'] = data.apply(lambda row: preprocess(str(row[0])), axis=1)
    data['text'] = data.apply(lambda row: str(row[0]), axis=1)
    print("Number of 1s in the data: ", len(data[data['label']==1]),"Number of 0s in the data: ", len(data[data['label']==0]))
    print("Total dataset size: ", len(data))

    X, y, vocabulary, vocabulary_inv = load_data(max_len, data)

    print("length of X, y after converting data into indices, labels: ", len(X), len(y))
    print("Vocabulary size is: ", len(list(vocabulary.keys())))


    embedding_dim = 768
    embeddings = 1 * np.random.randn(len(vocabulary) + 1, embedding_dim)
    embeddings[0] = 0

    count = 0
    for word, index in vocabulary.items():
        count += 1
        if count % 100 == 0:
            print("Finished encoding ", count, " words")
        embeddings[index] = bert_model.encode(word)

    # embed_f = './bert_embeddings'
    # e_f = open(embed_f, 'rb')
    # embeddings = pickle.load(e_f)
    # e_f.close()

    print("Vocabulary shape is ", embeddings.shape)

    X_train,X_test, Y_train, Y_test =  train_test_split(X, y,test_size = 0.15,random_state= 5)
    X_train,X_eval, Y_train, Y_eval =  train_test_split(X_train, Y_train,test_size = 0.15,random_state= 5)

    X_train = np.array(X_train, dtype=np.float)
    X_test = np.array(X_test, dtype=np.float)
    Y_train = np.array(Y_train, dtype=np.float)
    Y_test = np.array(Y_test, dtype=np.float)

    print("Validation dimensions ", len(X_eval), len(Y_eval))
    return X_train,X_test, X_eval, Y_train, Y_test, Y_eval, embeddings

class TestCallback(Callback):
    def __init__(self, x, y, output_file_name):
        self.x = x
        self.y = y
        self.output_file_name = output_file_name

    def on_epoch_end(self, epoch, logs={}):
        output = self.model.evaluate(self.x, self.y, verbose=0)
        fi = open("results/test_"+self.output_file_name, "a")
        for a in output:
            fi.write(str(a))
        fi.write("\n")
        print('\nTesting values: {}\n'.format(output))
        fi.close()

def main(length, rnn_dim, dropO1, seed, X_train,X_test, X_eval, Y_train, Y_test, Y_eval, embeddings):

    np.random.seed(seed)
    tf.compat.v1.set_random_seed(seed)
    random.seed(seed)
    max_len = length
    print("Starting new run- length: ", str(length), " rnn_dim: ", str(rnn_dim), " dropO1: ", dropO1, " seed: ", seed)
    # model, att_mod = c_bilstm(max_len, {'embed_mat': embeddings}, rnn_dim, att_dim, dropO1, dropO2, 'sigmoid', 1, rnn_type, num_cnn_filters, kernel_sizes)
    # model, att_mod = flat_fuse(max_len, rnn_dim, att_dim, [{'embed_mat':embeddings, 's_enc': 'rnn', 'm_id': '21'}], 'sen_enc_feats', dropO1, dropO2, 'sigmoid', 1, rnn_type, True, num_cnn_filters, 1, kernel_sizes)
        
    n_epoch = 15

    left_input = Input(shape=(length,))

    embedding_layer = Embedding(len(embeddings), 768, weights=[embeddings], input_length=max_len, trainable=True)

    encoded_left = embedding_layer(left_input)

    encoder_lstm = Bidirectional (LSTM (rnn_dim, return_sequences=False, dropout=dropO1),merge_mode='concat')
    first_out = encoder_lstm (encoded_left)

    # merged = Concatenate(axis=1)([first_out, second_out])

    output_layer1 = Dense(128, activation='relu')(first_out)
    output_layer2 = Dense(32, activation='relu')(output_layer1)
    output_layer3 = Dense(1, activation='sigmoid')(output_layer2)

    # model = Model(inputs=[left_input, right_input], outputs=malstm_distance)
    model = Model(left_input, output_layer3)
    model.compile('adam', loss='binary_crossentropy', metrics=['accuracy', get_f1, macro_f1, recall, precision, mcc, keras.metrics.TruePositives(name='tp'), keras.metrics.FalsePositives(name='fp'), keras.metrics.TrueNegatives(name='tn'), keras.metrics.FalseNegatives(name='fn')])

    print("Model summary:\n", model.summary())

    early_stopping = EarlyStopping(monitor="val_accuracy", mode='max',patience=4, verbose=True)

    training_start_time = time()

    output_file_name = "CNN" + str(length) + "_" + str(rnn_dim) + "_" + str(int(dropO1*100)) + "_"  + "_" + str(seed)
    fi = open('results/complete_results.csv', "a")
    fi.write("\nCurrent model is;" + output_file_name + "\n")
    fi.close()

    csv_logger = CSVLogger("results/complete_results.csv", append=True, separator=',')
    model.fit(X_train, Y_train, epochs=15, validation_data=(X_eval,Y_eval),callbacks = [early_stopping, csv_logger,ModelCheckpoint(filepath="models/"+output_file_name, save_best_only=True, monitor="val_accuracy", mode='max'), TestCallback(X_test, Y_test, output_file_name)])

    print("Training time finished.\n{} epochs in {}".format(10, datetime.timedelta(seconds=time()-training_start_time)))

    test_output = model.evaluate(X_test, Y_test, verbose = True)
    fi = open("results/test_results.csv", "a")
    fi.write("\nCurrent model is;" + output_file_name + "\n")
    temp = []
    for a in test_output:
        temp.append(str(a))
    fi.write(','.join(temp))
    fi.write("\n")
    fi.close()


X_train,X_test, X_eval, Y_train, Y_test, Y_eval, embeddings = init(42, 150)


seeds = [5, 21, 42]
lengths = [150]
dropO1s = [0.25]
rnn_dims = [150]

for seed in seeds:
    for length in lengths:
        for dropO1 in dropO1s:
            for rnn_dim in rnn_dims:
                    main(length, rnn_dim, dropO1, seed, X_train,X_test, X_eval, Y_train, Y_test, Y_eval, embeddings)




