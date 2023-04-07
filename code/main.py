import pysam as ps
import numpy as np
# import collections
# import matplotlib.pyplot as plt
# from numpy.core.defchararray import array, center
import pysam as ps
import numpy as np
# from collections import defaultdict
# from scipy.ndimage.measurements import label, standard_deviation
# from scipy.stats.mstats_basic import kstest, normaltest
# from sklearn.cluster import KMeans
# import sys
# from scipy.signal import savgol_filter
# import math
# from subprocess import call
# import os.path
from utils import Gene, TSS, Point
# from scipy import stats
# from sklearn import svm
# import sympy
# import math
# from math import e
import random
from tensorflow.keras import Model
from tensorflow.keras.models import load_model
import numpy as np
# import os
# import pandas as pd
import tensorflow as tf
from tensorflow import keras
# from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras import backend as K
from sklearn.model_selection import KFold
from tensorflow.keras.layers import Input,Conv2D,Activation,Dense,Lambda,Flatten,Embedding,PReLU,BatchNormalization,Bidirectional,LSTM
from tensorflow.keras.models import Model
from tensorflow.keras.utils import to_categorical
# from tensorflow.keras import layers
# from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
# from sklearn.model_selection import KFold
from tensorflow.keras import backend as K
# import copy
from sklearn.utils import shuffle
# import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"
def set_seef(seed):
    np.random.seed(seed)
    random.seed(seed)
    tf.random.set_seed(seed)
set_seef(random.randint(1,10000))
dic = {
    'gene':'../data/gene/GRCh37.gene.bed',
    'non_gene':'../data/gene/non_gene_1234567.bed',
    'non_gene_4':'../data/gene/non_gene_4.bed',
    'non_gene_2':'../data/gene/non_gene_2.bed',
    # 'fasta':'/home/jiay/Desktop/hg19/hg19.fa',
    'bam1234567':'../data/051_1234567.bam',
    'TSS_low':'../data/gene/low_expressed.bed',
    'TSS_HK':'../data/gene/HK.bed',
    'TSS_silent':'../data/gene/silent_gene_TSS.bed',
    'ATAC_Bcell':'../data/gene/ATAC_Bcell.bed',
    'ATAC_Brain':'../data/gene/ATAC_Brain.bed',
    'ATAC_hema':'../data/gene/ATAC_hema.bed',
    'model_save':'../model'
    }
# TSS_HK = []
# with open(dic['TSS_HK'],'r') as f:
#     for line in f:
#         ll = line.strip().split('\t')
#         if ll[0] in ['2','3','4','5','6','7']:
#         # if ll[0] in ['1','2','3']:
#             TSS_HK.append(TSS(ll[0], int(int(ll[1])+1000)))
# # with open(dic['ATAC_hema'],'r') as f:
# #     for line in f:
# #         ll = line.strip().split('\t')
# #         if ll[0] in ['1']:
# #             TSS_HK.append(TSS(ll[0], int((int(ll[1])+int(ll[2]))/2)))
# # with open(dic['ATAC_Bcell'],'r') as f:
# #     for line in f:
# #         ll = line.strip().split('\t')
# #         if ll[0] in ['3']:
# #             TSS_HK.append(TSS(ll[0], int(int(ll[1])+1000)))        


# TSS_NonGene = []
# with open(dic['non_gene'],'r') as f:
#     for line in f:
#         ll = line.strip().split('\t')
#         if ll[0][-1] in ['2','3','4','5','6','7']:
#         # if ll[0][-1] in ['1','2','3']:
#             TSS_NonGene.append(TSS(ll[0], int(ll[1])+int(1000)))
# # with open(dic['ATAC_Bcell'],'r') as f:
# #     for line in f:
# #         ll = line.strip().split('\t')
# #         if ll[0] in ['1','2','3'] and ll[2] == '0':
# #             TSS_NonGene.append(TSS(ll[0],int(ll[1])))

# bamfile = ps.AlignmentFile(dic['bam1234567'],'rb')

# TSS_NonGene = shuffle(TSS_NonGene)[:len(TSS_HK)]

# TSSes_x = TSS_HK + TSS_NonGene
# TSSes_y = [1]*len(TSS_HK) + [0]*len(TSS_NonGene)
# perm = random.sample(range(len(TSSes_x)),len(TSSes_x))
# TSSes_x = np.array(TSSes_x)
# TSSes_y = np.array(TSSes_y)
# TSSes_x = TSSes_x[perm[:len(perm)]]
# TSSes_y = TSSes_y[perm[:len(perm)]]
# labels = TSSes_y
# up = 1000
# down = 1000

# '''cnn_input'''
# raw_data = []
# for j, tss in enumerate(TSSes_x):
#     chrom = tss.chrom
#     start = tss.pos - up
#     end = tss.pos + down
# #    distribution_matrix = np.zeros((int(up+down), 200), dtype=int)
#     distribution_matrix = np.zeros((200,200),dtype=int)
#     for r in bamfile.fetch(chrom[-1], start-500, end + 500):
#         if (not r.is_reverse) and (not r.is_unmapped) and (not r.mate_is_unmapped) and r.mate_is_reverse and 50 < abs(r.isize) < 250:
#             if r.reference_start + abs(r.isize) < start:
#                 continue
#             if r.reference_start >= end:
#                 continue
#             if r.reference_start < start:
#                 continue
#             if r.reference_start + abs(r.isize) > end:
#                 continue
#             ss = max(0, r.reference_start - start)
#             relative_isize = abs(r.isize)-50
#             distribution_matrix[ss//10,relative_isize] += 1
#     raw_data.append(distribution_matrix)
# raw_data = np.array(raw_data)
# cnn_x = []
# for mat in raw_data:
#     cnn_x.append(mat)
# cnn_x = np.array(cnn_x)
cnn_x = np.load('../datas/train_hk27_cnn_x.npy')
lstm_x = np.load('../datas/train_hk27_lstm_x.npy')
labels = np.load('../datas/train_hk27_y.npy')
cnn_x, lstm_x, labels = shuffle(cnn_x, lstm_x, labels)
# '''lstm input'''
# feature_matrix = []
# for j, tss in enumerate(TSSes_x):
#     chrom = tss.chrom
#     start = tss.pos - up
#     end = tss.pos + down
#     up_end = np.zeros(up+down, dtype= int)
#     down_end = np.zeros(up+down, dtype= int)
#     long = np.zeros(up+down, dtype= int)
#     short = np.zeros(up+down, dtype= int)
#     cov = np.zeros(up+down, dtype= int)
#     wps = np.zeros(up+down, dtype=float)
#     win = 120
#     for r in bamfile.fetch(chrom[-1], start-500, end + 500):
#         if (not r.is_reverse) and (not r.is_unmapped) and (not r.mate_is_unmapped) and r.mate_is_reverse:
#             if r.reference_start + abs(r.isize) < start:
#                 continue
#             if r.reference_start >= end:
#                 continue
#             ss = r.reference_start - start
#             ee = r.reference_start - start + abs(r.isize)
#             if ss >= 0:
#                 up_end[ss] += 1
#             else:
#                 ss = 0
#             if ee < end - start:
#                 down_end[ee] += 1
#             else:
#                 ee = end - start
#             for i in range(ss, ee):
#                 cov[i] += 1
#             if 200 >= abs(r.isize) > 130:
#                 for i in range(ss, ee):
#                     long[i] += 1
#             if abs(r.isize) <= 130:
#                 for i in range(ss, ee):
#                     short[i] += 1
#             # wps_total
#             region1 = int(max(0, ss + win/2))
#             region2 = int(min(ee - win/2, end-start))
#             i = region1
#             while i < region2:
#                 wps[i] += 1
#                 i = i+1
#             # wps_part
#             region1 = int(max(0, ss - win/2))
#             region2 = int(min(end-start, ss + win/2))
#             i = region1
#             while i < region2:
#                 wps[i] -= 1
#                 i = i + 1
#             # wps_part
#             region1 = int(max(ee - win/2, 0))
#             region2 = int(min(ee + win/2, end-start))
#             i = region1
#             while i < region2:
#                 wps[i] -= 1
#                 i = i+1
#     k = 0
#     win = 40
#     feature_win = np.zeros((int((up+down)/win), 4), dtype= int)
#     while k < (up+down)/win:
#         ss = k * win
#         ee = k * win + win
#         ff = []
#         ff.append(int(round(np.mean(cov[ss:ee]))))
#         ff.append(int(round(np.mean(long[ss:ee]-short[ss:ee]))))
#         ff.append(int(round(np.sum(abs(up_end[ss:ee]-down_end[ss:ee])))))
#         ff.append(int(round(np.mean(wps[ss:ee]))))
#         feature_win[k] = np.array(ff)
#         k = k + 1
#     feature_matrix.append(feature_win)
# feature_matrix = np.array(feature_matrix)
# lstm_x = []
# for mat in feature_matrix:
#     lstm_x.append(mat)
# lstm_x = np.array(lstm_x)

'''create OCRFinder-model function'''
from tensorflow.keras.layers import Input, Embedding, LSTM, Dense, concatenate, Convolution1D, MaxPooling1D, Flatten, Bidirectional,Dropout, AveragePooling1D
from tensorflow.keras.models import Model
from tensorflow import keras
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
def create_model():
    cnn_input = Input(shape=(cnn_x.shape[1], cnn_x.shape[2]), name='cnn_input')
    lstm_input = Input(shape=(lstm_x.shape[1], lstm_x.shape[2]), name='lstm_input')
    conv1 = Convolution1D(filters=64, kernel_size=2, strides=2, activation='relu', padding='same',name='conv1')(cnn_input)
    conv1 = AveragePooling1D(pool_size=2, strides=2)(conv1)
    cnn_output = Model(inputs=cnn_input, outputs=conv1)
    main_input = concatenate([cnn_output.output, lstm_input])
    lstm_out = Bidirectional(LSTM(50, return_sequences=True),name='0')(main_input)
    conv = Convolution1D(filters=100, kernel_size=3, activation='relu',strides=1,padding='same',name='1')(lstm_out)
    pool = MaxPooling1D(pool_size=2, strides=2,name='2')(conv)
    drop = Dropout(0.2)(pool)
    flatten = Flatten()(drop)
    dense = Dense(300, activation='relu', kernel_regularizer=None, bias_regularizer=None,name='4')(flatten)
    drop = Dropout(0.2)(dense)
    main_output = Dense(1, activation='sigmoid', kernel_regularizer=None, bias_regularizer=None,name='6')(drop)
    model = Model(inputs=[cnn_output.input, lstm_input], outputs=main_output)
    return model

# '''create cnn model function'''
# from tensorflow.keras.layers import Input, Embedding, LSTM, Dense, concatenate, Convolution1D, MaxPooling1D, Flatten, Bidirectional,Dropout, AveragePooling1D
# from tensorflow.keras.models import Model
# from tensorflow import keras
# from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
# def create_model():
#     cnn_input = Input(shape=(cnn_x.shape[1], cnn_x.shape[2]), name='cnn_input')
#     conv1 = Convolution1D(filters=64, kernel_size=2, strides=2, activation='relu', padding='same',name='conv1')(cnn_input)
#     conv1 = AveragePooling1D(pool_size=2, strides=2)(conv1)
#     conv = Convolution1D(filters=100, kernel_size=3, activation='relu',strides=1,padding='same',name='1')(conv1)
#     pool = MaxPooling1D(pool_size=2, strides=2,name='2')(conv)
#     drop = Dropout(0.2)(pool)
#     flatten = Flatten()(drop)
#     dense = Dense(300, activation='relu', kernel_regularizer=None, bias_regularizer=None,name='4')(flatten)
#     drop = Dropout(0.2)(dense)
#     main_output = Dense(1, activation='sigmoid', kernel_regularizer=None, bias_regularizer=None,name='6')(drop)
#     model = Model(inputs=cnn_input, outputs=main_output)
#     return model
# ''' create conv-lstm model function'''
# from tensorflow.keras.layers import Input, Embedding, LSTM, Dense, concatenate, Convolution1D, MaxPooling1D, Flatten, Bidirectional,Dropout, AveragePooling1D
# from tensorflow.keras.models import Model
# from tensorflow import keras
# from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
# def create_model():
#     cnn_input = Input(shape=(cnn_x.shape[1], cnn_x.shape[2]), name='cnn_input')
#     conv1 = Convolution1D(filters=64, kernel_size=2, strides=2, activation='relu', padding='same',name='conv1')(cnn_input)
#     conv1 = AveragePooling1D(pool_size=2, strides=2)(conv1)
#     lstm_out = Bidirectional(LSTM(50, return_sequences=True),name='0')(conv1)
#     conv = Convolution1D(filters=100, kernel_size=3, activation='relu',strides=1,padding='same',name='1')(lstm_out)
#     pool = MaxPooling1D(pool_size=2, strides=2,name='2')(conv)
#     drop = Dropout(0.2)(pool)
#     flatten = Flatten()(drop)
#     dense = Dense(300, activation='relu', kernel_regularizer=None, bias_regularizer=None,name='4')(flatten)
#     drop = Dropout(0.2)(dense)
#     main_output = Dense(1, activation='sigmoid', kernel_regularizer=None, bias_regularizer=None,name='6')(drop)
#     model = Model(inputs=cnn_input, outputs=main_output)
#     return model
# '''create lstm model function'''
# from tensorflow.keras.layers import Input, Embedding, LSTM, Dense, concatenate, Convolution1D, MaxPooling1D, Flatten, Bidirectional,Dropout, AveragePooling1D
# from tensorflow.keras.models import Model
# from tensorflow import keras
# from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
# def create_model():
#     lstm_input = Input(shape=(lstm_x.shape[1], lstm_x.shape[2]), name='lstm_input')
#     lstm_out = Bidirectional(LSTM(50, return_sequences=True),name='0')(lstm_input)
#     conv = Convolution1D(filters=100, kernel_size=3, activation='relu',strides=1,padding='same',name='1')(lstm_out)
#     pool = MaxPooling1D(pool_size=2, strides=2,name='2')(conv)
#     drop = Dropout(0.2)(pool)
#     flatten = Flatten()(drop)
#     dense = Dense(300, activation='relu', kernel_regularizer=None, bias_regularizer=None,name='4')(flatten)
#     drop = Dropout(0.2)(dense)
#     main_output = Dense(1, activation='sigmoid', kernel_regularizer=None, bias_regularizer=None,name='6')(drop)
#     model = Model(inputs=lstm_input, outputs=main_output)
#     return model

nb_epochs = 150
batchsize = 128
# '''base model'''
# for i in range(1,6):
#     set_seef(random.randint(1,10000))
#     base_model = create_model() 
#     adam = keras.optimizers.Adam(lr=1e-4, decay=1e-4)
#     base_model.compile(optimizer=adam, loss='binary_crossentropy', metrics=['accuracy'])
#     history = base_model.fit(lstm_x, labels, epochs=nb_epochs, batch_size=batchsize,shuffle=True, verbose=2)
#     # history = base_model.fit(cnn_x, labels, epochs=nb_epochs, batch_size=batchsize,shuffle=True, verbose=2)
#     # history = base_model.fit([cnn_x, lstm_x], labels, epochs=nb_epochs, batch_size=batchsize,shuffle=True, verbose=2)
#     base_model.save('../addition_model/lstm_base'+str(i)+'_hk_150.h5')    
# exit(0)
''' OCRFinder training framework '''
loss_object = tf.keras.losses.BinaryCrossentropy(from_logits = False, reduction = tf.keras.losses.Reduction.NONE)
def get_loss0(model, x, y):
    pred_y = model(x, training=False)
    return loss_object(tf.reshape(y, (-1,1)), tf.reshape(pred_y,(-1,1)))
def focal_loss(y_true, y_pred):
    gamma = 2.0
    alpha = 0.5
    pt_1 = tf.where(tf.equal(y_true, 1), y_pred, tf.ones_like(y_pred))
    pt_0 = tf.where(tf.equal(y_true, 0), y_pred, tf.zeros_like(y_pred))
    return -K.sum(alpha * K.pow(1. - pt_1, gamma) * K.log(pt_1))-K.sum((1-alpha) * K.pow( pt_0, gamma) * K.log(1. - pt_0))


def CE_L1_loss(y_true, y_pred):
    weight = 2
    # ww = tf.where(tf.equal(y_true, 1), y_pred*K.log(y_pred), (1-y_pred)*K.log(1-y_pred))
    loss = K.binary_crossentropy(y_true, y_pred) + weight *K.abs((y_pred-y_true))
    return loss
def get_loss1(model, x, y):
    pred_y = model(x, training=False)
    return loss_object(y_true = y, y_pred= np.array(pred_y).reshape(-1,1))
def get_loss(y_t, y_p):
    return loss_object(tf.reshape(y_t,(-1,1)), tf.reshape(y_p,(-1,1)))
from tqdm import tqdm
def Sharpen(ypp, t=0.4):
    return K.exp(ypp/t) /(K.exp(ypp/t)+K.exp((1-ypp)/t))
l1_object = tf.keras.losses.BinaryCrossentropy(from_logits = False, reduction = tf.keras.losses.Reduction.SUM_OVER_BATCH_SIZE)     
def loss_fn(yt, yp, ypp, ypp1, ypp2, gamma=1, sharpen=True, semi=True):
    l1 = tf.losses.binary_crossentropy(tf.reshape(yt,(-1,1)), tf.reshape(yp, (-1,1)))
    if sharpen and semi:
        # l2 = K.mean(K.square(ypp - (ypp1+ypp2)/2), axis=-1)
        # label0 = (ypp1+ypp2)/2
        # labeli = K.exp(label0/0.3)/(K.exp(label0/0.3)+K.exp((1-label0)/0.3))
        # l2 = K.square(ypp - labeli)
        labeli = (Sharpen(ypp1) + Sharpen(ypp2))/2
        l2 = K.square(ypp - labeli)
    elif semi:
        p1 = (ypp1 + ypp2)/2
        p2 = 1 - (ypp1 + ypp2)/2
        p_s = p1/(p1+p2)
        l2 = K.square(ypp - p_s)
    else:
        l2 = 0
    return K.mean(l1)  + gamma * K.mean(l2, axis = -1)
def loss_fn_new(yt, yp, ypp, yppp, gamma=0.5):
    l1 = tf.losses.binary_crossentropy(tf.reshape(yt,(-1,1)), tf.reshape(yp, (-1,1)))
    l2 = K.square(ypp - yppp)
    return K.mean(l1) + gamma * K.mean(l2, axis = -1)

# for vvv in range(1, 6):
#     set_seef(random.randint(1,10000))
#     pre_epochs = 5
#     lam = 0.3
#     model_a = create_model()
#     model_b = create_model()

#     model_a.compile(optimizer=keras.optimizers.Adam(lr=1e-4, decay=1e-4), loss=CE_L1_loss, metrics=['accuracy'])
#     model_b.compile(optimizer=keras.optimizers.Adam(lr=1e-4, decay=1e-4), loss=CE_L1_loss, metrics=['accuracy'])
#     # model_a.compile(optimizer=keras.optimizers.Adam(lr=1e-4, decay=1e-4), loss='binary_crossentropy', metrics=['accuracy'])
#     # model_b.compile(optimizer=keras.optimizers.Adam(lr=1e-4, decay=1e-4), loss='binary_crossentropy', metrics=['accuracy'])


#     input_1, input_2, y_train = shuffle(np.array(cnn_x,dtype='float32'), np.array(lstm_x,dtype='float32'), np.array(labels,dtype='float32'))
#     # input_1, y_train = shuffle(np.array(cnn_x,dtype='float32'), np.array(labels,dtype='float32'))

#     model_a.fit([input_1, input_2], y_train, epochs=5, batch_size=batchsize, shuffle=True, verbose=0)
#     model_b.fit([input_1, input_2], y_train, epochs=5, batch_size=batchsize, shuffle=True, verbose=0)
#     # model_a.fit(input_1, y_train, epochs=5, batch_size=batchsize, shuffle=True, verbose=0)
#     # model_b.fit(input_1, y_train, epochs=5, batch_size=batchsize, shuffle=True, verbose=0)
#     # model_a.fit([input_1, input_2], y_train, epochs=pre_epochs, batch_size=batchsize, shuffle=True,verbose=0)
#     # model_b.fit([input_1, input_2], y_train, epochs=pre_epochs, batch_size=batchsize, shuffle=True,verbose=0)

#     nb_batch = int(np.ceil(len(y_train) / batchsize))
#     optimizer_a = keras.optimizers.Adam(learning_rate=1e-4)
#     optimizer_b = keras.optimizers.Adam(learning_rate=1e-4)
#     pre_a = model_a([input_1, input_2],training=False).numpy()
#     pre_b = model_b([input_1, input_2],training=False).numpy()
#     # pre_a = model_a(input_1,training=False).numpy()
#     # pre_b = model_b(input_1,training=False).numpy()
#     loss_a = get_loss(y_train, pre_a).numpy()
#     loss_b = get_loss(y_train, pre_b).numpy()
#     for ep in tqdm(range(nb_epochs)):
#         # input_1, input_2, y_train = shuffle(input_1, input_2, y_train)
#         input_1, y_train = shuffle(input_1, y_train)
#         p_a = model_a([input_1, input_2],training=False)
#         # p_a = model_a(input_1,training=False)
#         l_a = get_loss(y_train, p_a.numpy()).numpy()
#         pre_a = lam * p_a.numpy() + (1-lam) * pre_a
#         loss_a = lam * l_a + (1-lam) * loss_a
#         p_b = model_b([input_1, input_2],training=False)
#         # p_b = model_b(input_1,training=False)
#         l_b = get_loss(y_train, p_b.numpy()).numpy()
#         pre_b = lam * p_b.numpy() + (1-lam) * pre_b
#         loss_b = lam * l_b + (1-lam) * loss_b

#         mean_a = np.mean(loss_a)
#         data_ids_a = []
#         semi_ids_a = []
#         for id, ll in enumerate(loss_a):
#             if ll < mean_a:
#                 data_ids_a.append(id)
#             else:
#                 if (pre_a[id] -0.5) * (pre_b[id]-0.5) > 0:
#                     semi_ids_a.append(id)
#         mean_b = np.mean(loss_b)             
#         data_ids_b = []
#         semi_ids_b = []
#         for id, ll in enumerate(loss_b):
#             if ll < mean_b:
#                 data_ids_b.append(id)
#             else:
#                 if (pre_a[id]-0.5)*(pre_b[id]-0.5) > 0:
#                     semi_ids_b.append(id)
#         perm = random.sample(range(len(y_train)),len(y_train))
#         for nb in range(nb_batch):
#             start = nb * batchsize
#             end = min((nb + 1) * batchsize, len(y_train))
#             # x_batch_train, y_batch_train = x_train[start:end], y_train[start:end]
#             tr_a_ids = []
#             ex_a_ids = []
#             tr_b_ids = []
#             ex_b_ids = []
#             # for id in range(start,end):
#             for id in perm[start:end]:
#                 if id in data_ids_a:
#                     tr_a_ids.append(id)
#                 elif id in semi_ids_a:
#                     ex_a_ids.append(id)
#                 if id in data_ids_b:
#                     tr_b_ids.append(id)
#                 elif id in semi_ids_b:
#                     ex_b_ids.append(id)

#             with tf.GradientTape() as tape_a:
#                 logits_a = model_a([input_1[tr_b_ids], input_2[tr_b_ids]], training=True)
#                 log_a = model_a([input_1[ex_b_ids], input_2[ex_b_ids]], training=True)
#                 # logits_a = model_a(input_1[tr_b_ids], training=True)
#                 # log_a = model_a(input_1[ex_b_ids], training=True)
#                 # loss_value_a = loss_fn_new(y_train[tr_b_ids], logits_a, log_a, pre_sharpen[ex_b_ids])
#                 loss_value_a = loss_fn(y_train[tr_b_ids], logits_a, log_a, pre_a[ex_b_ids], pre_b[ex_b_ids],0.5,True)
#             grads_a = tape_a.gradient(loss_value_a, model_a.trainable_weights)
#             optimizer_a.apply_gradients(zip(grads_a, model_a.trainable_weights))
#             with tf.GradientTape() as tape_b:
#                 logits_b = model_b([input_1[tr_a_ids], input_2[tr_a_ids]], training=True)
#                 log_b = model_b([input_1[ex_a_ids], input_2[ex_a_ids]], training=True)
#                 # logits_b = model_b(input_1[tr_a_ids], training=True)
#                 # log_b = model_b(input_1[ex_a_ids], training=True)
#                 # loss_value_b = loss_fn_new(y_train[tr_a_ids], logits_b, log_b, pre_sharpen[ex_a_ids])
#                 loss_value_b = loss_fn(y_train[tr_a_ids], logits_b, log_b, pre_a[ex_a_ids], pre_b[ex_a_ids], 0.5, True)
#             grads_b = tape_b.gradient(loss_value_b, model_b.trainable_weights)
#             optimizer_b.apply_gradients(zip(grads_b, model_b.trainable_weights))
#     print('vv=%s'%vvv)
#     model_a.save('../addition_model/model_a_convlstm_OCRFinder'+str(vvv)+'_hk_150.h5')
#     # model_b.save('../addition_model/model_b_convlstm_OCRFinder'+str(vvv)+'_hk_150.h5')

for vvv in range(1, 6):
    set_seef(random.randint(1,10000))
    pre_epochs = 5
    lam = 0.3
    model_a = create_model()

    model_a.compile(optimizer=keras.optimizers.Adam(lr=1e-4, decay=1e-4), loss=CE_L1_loss, metrics=['accuracy'])
    # model_a.compile(optimizer=keras.optimizers.Adam(lr=1e-4, decay=1e-4), loss='binary_crossentropy', metrics=['accuracy'])
    # model_b.compile(optimizer=keras.optimizers.Adam(lr=1e-4, decay=1e-4), loss='binary_crossentropy', metrics=['accuracy'])


    input_1, input_2, y_train = shuffle(np.array(cnn_x,dtype='float32'), np.array(lstm_x,dtype='float32'), np.array(labels,dtype='float32'))
    # input_1, y_train = shuffle(np.array(cnn_x,dtype='float32'), np.array(labels,dtype='float32'))

    model_a.fit([input_1, input_2], y_train, epochs=5, batch_size=batchsize, shuffle=True, verbose=0)
    # model_a.fit(input_1, y_train, epochs=5, batch_size=batchsize, shuffle=True, verbose=0)
    # model_b.fit(input_1, y_train, epochs=5, batch_size=batchsize, shuffle=True, verbose=0)
    # model_a.fit([input_1, input_2], y_train, epochs=pre_epochs, batch_size=batchsize, shuffle=True,verbose=0)
    # model_b.fit([input_1, input_2], y_train, epochs=pre_epochs, batch_size=batchsize, shuffle=True,verbose=0)

    nb_batch = int(np.ceil(len(y_train) / batchsize))
    optimizer_a = keras.optimizers.Adam(learning_rate=1e-4)
    pre_a = model_a([input_1, input_2],training=False).numpy()
    # pre_a = model_a(input_1,training=False).numpy()
    # pre_b = model_b(input_1,training=False).numpy()
    loss_a = get_loss(y_train, pre_a).numpy()
    for ep in tqdm(range(nb_epochs)):
        input_1, input_2, y_train = shuffle(input_1, input_2, y_train)
        # input_1, y_train = shuffle(input_1, y_train)
        p_a = model_a([input_1, input_2],training=False)
        # p_a = model_a(input_1,training=False)
        l_a = get_loss(y_train, p_a.numpy()).numpy()
        pre_a = lam * p_a.numpy() + (1-lam) * pre_a
        loss_a = lam * l_a + (1-lam) * loss_a

        mean_a = np.mean(loss_a)
        data_ids_a = []
        semi_ids_a = []
        for id, ll in enumerate(loss_a):
            if ll < mean_a:
                data_ids_a.append(id)
            else:
                if (pre_a[id] -0.5) * (pre_a[id]-0.5) > 0:
                    semi_ids_a.append(id)

        perm = random.sample(range(len(y_train)),len(y_train))
        for nb in range(nb_batch):
            start = nb * batchsize
            end = min((nb + 1) * batchsize, len(y_train))
            tr_a_ids = []
            ex_a_ids = []

            # for id in range(start,end):
            for id in perm[start:end]:
                if id in data_ids_a:
                    tr_a_ids.append(id)
                elif id in semi_ids_a:
                    ex_a_ids.append(id)

            with tf.GradientTape() as tape_a:
                logits_a = model_a([input_1[tr_a_ids], input_2[tr_a_ids]], training=True)
                log_a = model_a([input_1[ex_a_ids], input_2[ex_a_ids]], training=True)
                loss_value_a = loss_fn(y_train[tr_a_ids], logits_a, log_a, pre_a[ex_a_ids], pre_a[ex_a_ids],0.5,True)
            grads_a = tape_a.gradient(loss_value_a, model_a.trainable_weights)
            optimizer_a.apply_gradients(zip(grads_a, model_a.trainable_weights))

    print('vv=%s'%vvv)
    model_a.save('../addition_model/model_OCRFinder_nonco'+str(vvv)+'_hk_150.h5')
    # model_b.save('../addition_model/model_b_convlstm_OCRFinder'+str(vvv)+'_hk_150.h5')

# exit(0)
# for vvv in range(1, 4):
#     set_seef(random.randint(1,10000))
#     pre_epochs = 5
#     lam = 1
#     model_a = create_model()
#     model_b = create_model()

#     model_a.compile(optimizer=keras.optimizers.Adam(lr=1e-4, decay=1e-4), loss=CE_L1_loss, metrics=['accuracy'])
#     model_b.compile(optimizer=keras.optimizers.Adam(lr=1e-4, decay=1e-4), loss=CE_L1_loss, metrics=['accuracy'])


#     input_1, input_2, y_train = shuffle(np.array(cnn_x,dtype='float32'), np.array(lstm_x,dtype='float32'), np.array(labels,dtype='float32'))
#     model_a.fit([input_1, input_2], y_train, epochs=5, batch_size=batchsize, shuffle=True, verbose=0)
#     model_b.fit([input_1, input_2], y_train, epochs=5, batch_size=batchsize, shuffle=True, verbose=0)

#     # model_a.fit([input_1, input_2], y_train, epochs=pre_epochs, batch_size=batchsize, shuffle=True,verbose=0)
#     # model_b.fit([input_1, input_2], y_train, epochs=pre_epochs, batch_size=batchsize, shuffle=True,verbose=0)

#     nb_batch = int(np.ceil(len(y_train) / batchsize))
#     optimizer_a = keras.optimizers.Adam(learning_rate=1e-4)
#     optimizer_b = keras.optimizers.Adam(learning_rate=1e-4)
#     pre_a = model_a([input_1, input_2],training=False).numpy()
#     pre_b = model_b([input_1, input_2],training=False).numpy()
#     loss_a = get_loss(y_train, pre_a).numpy()
#     loss_b = get_loss(y_train, pre_b).numpy()
#     for ep in tqdm(range(nb_epochs)):
#         # input_1, input_2, y_train = shuffle(input_1, input_2, y_train)
#         p_a = model_a([input_1, input_2],training=False)
#         l_a = get_loss(y_train, p_a.numpy()).numpy()
#         pre_a = lam * p_a.numpy() + (1-lam) * pre_a
#         loss_a= lam * l_a + (1-lam) * loss_a
#         p_b = model_b([input_1, input_2],training=False)
#         l_b = get_loss(y_train, p_b.numpy()).numpy()
#         pre_b = lam * p_b.numpy() + (1-lam) * pre_b
#         loss_b = lam * l_b + (1-lam) * loss_b

#         mean_a = np.mean(loss_a)
#         data_ids_a = []
#         semi_ids_a = []
#         for id, ll in enumerate(loss_a):
#             if ll < mean_a:
#                 data_ids_a.append(id)
#             else:
#                 if (pre_a[id] -0.5) * (pre_b[id]-0.5) > 0:
#                     semi_ids_a.append(id)
#         mean_b = np.mean(loss_b)             
#         data_ids_b = []
#         semi_ids_b = []
#         for id, ll in enumerate(loss_b):
#             if ll < mean_b:
#                 data_ids_b.append(id)
#             else:
#                 if (pre_a[id]-0.5)*(pre_b[id]-0.5) > 0:
#                     semi_ids_b.append(id)
#         perm = random.sample(range(len(y_train)),len(y_train))
#         for nb in range(nb_batch):
#             start = nb * batchsize
#             end = min((nb + 1) * batchsize, len(y_train))
#             # x_batch_train, y_batch_train = x_train[start:end], y_train[start:end]
#             tr_a_ids = []
#             ex_a_ids = []
#             tr_b_ids = []
#             ex_b_ids = []
#             # for id in range(start,end):
#             for id in perm[start:end]:
#                 if id in data_ids_a:
#                     tr_a_ids.append(id)
#                 elif id in semi_ids_a:
#                     ex_a_ids.append(id)
#                 if id in data_ids_b:
#                     tr_b_ids.append(id)
#                 elif id in semi_ids_b:
#                     ex_b_ids.append(id)
#             with tf.GradientTape() as tape_a:
#                 logits_a = model_a([input_1[tr_b_ids], input_2[tr_b_ids]], training=True)
#                 log_a = model_a([input_1[ex_b_ids], input_2[ex_b_ids]], training=True)
#                 # loss_value_a = loss_fn_new(y_train[tr_b_ids], logits_a, log_a, pre_sharpen[ex_b_ids])
#                 loss_value_a = loss_fn(y_train[tr_b_ids], logits_a, log_a, pre_a[ex_b_ids], pre_b[ex_b_ids],0.5,False)
#             grads_a = tape_a.gradient(loss_value_a, model_a.trainable_weights)
#             optimizer_a.apply_gradients(zip(grads_a, model_a.trainable_weights))
#             with tf.GradientTape() as tape_b:
#                 logits_b = model_b([input_1[tr_a_ids], input_2[tr_a_ids]], training=True)
#                 log_b = model_b([input_1[ex_a_ids], input_2[ex_a_ids]], training=True)
#                 # loss_value_b = loss_fn_new(y_train[tr_a_ids], logits_b, log_b, pre_sharpen[ex_a_ids])
#                 loss_value_b = loss_fn(y_train[tr_a_ids], logits_b, log_b, pre_a[ex_a_ids], pre_b[ex_a_ids],0.5,False)
#             grads_b = tape_b.gradient(loss_value_b, model_b.trainable_weights)
#             optimizer_b.apply_gradients(zip(grads_b, model_b.trainable_weights))
#     print('vv=%s'%vvv)
#     model_a.save('../addition_model/model_a_OCRFinde_ensemble'+str(vvv)+'_hk_150.h5')
#     model_b.save('../addition_model/model_b_OCRFinder_ensemble'+str(vvv)+'_hk_150.h5')
