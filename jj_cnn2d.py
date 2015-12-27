# -*- coding: utf-8 -*-
'''

dstc2 text classifier cnn2d

@author: Jeongpil Lee (koreanfeel@gmail.com)

 
'''

import os
import json
import numpy as np
import nltk
from collections import OrderedDict
from sklearn import svm
import text_analizer
from sklearn.cross_validation import train_test_split
from numpy import linalg as LA
from keras.preprocessing import sequence
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D, Convolution1D, MaxPooling1D
from keras.layers.recurrent import LSTM, GRU, SimpleRNN
from keras.utils import np_utils
from keras.layers.embeddings import WordContextProduct, Embedding
from keras.layers.recurrent import LSTM

DATA_FILE = './data-train.json'
VECTOR_FILE = './glove.6B.50d.txt'

max_features = 5000
maxlen = 50
batch_size = 16
embedding_dims = 50
nb_filters = 100
hidden_dims = 100
nb_epoch = 100
# size of pooling area for max pooling
nb_pool = 2
# convolution kernel size
nb_conv = 3

if __name__ == '__main__':

    transcript = []  # 사용자의 대화 텍스트를 저장할 리스트
    acts_map = []
    train_X = []
    train_y = []
    max_word_len = 0

    ''' txt로부터 vector 가져오기 '''
    ta = text_analizer.TextAnalyzer(VECTOR_FILE)
    #     print(ta.txt2vectors("I'm a sogang university student."))

    ''' json 파일에서 필요한 정보 읽어오기 '''
    with open(DATA_FILE) as f:
        jData = json.load(f)

        for jSession in jData:
            jTurns = jSession.get('turns')
            for jTurn in jTurns:
                text = jTurn.get('user').get('transcript')

                ''' 가장 긴 word 수를 구함 '''
                word_len = len(text.split())
                if len(text.split()) > max_word_len:
                    max_word_len = word_len

        for jSession in jData:
            jTurns = jSession.get('turns')
            for jTurn in jTurns:
                #                 transcript.append(jTurn.get('system').get('transcript'))
                text = jTurn.get('user').get('transcript')
                transcript.append(text)

                ''' dialog act만 뽑아냄 '''
                dialog_acts = jTurn.get('user').get('dialog-acts')

                #! act + slot + val 값 모두를 키로 함
                acts_slot_val_list = []  # 한 턴에 act가 여러개인 경우를 처리하기 위한 리스트

                # for dialog_act in dialog_acts:
                    # if dialog_act['act'] not in acts:
                        # acts.append(dialog_act['act'])

                #! act + slot + value 모두를 더한 문자열을 구성
                for dialog_act in dialog_acts:
                    act_slot_val = str(dialog_act['act']) + str(dialog_act['slots'])
                    if act_slot_val not in acts_slot_val_list:
                        acts_slot_val_list.append(act_slot_val)
                        print(act_slot_val)
					

                acts_slot_val_list.sort()

                act_str = '-'.join(acts_slot_val_list)

                if act_str not in acts_map:
                    acts_map.append(act_str)

                vector_items = ta.txt2vectors(text)  # 이 부분에서 text를 vector로 변환함



                text_vectors = np.zeros(shape=(max_word_len, 50), dtype=float)

                # print(vector_items)

                i = 0

                for word in vector_items:
                    # print(vector_items[word])
                    text_vectors[i] = vector_items[word]
                    i += 1





                channel = []
                channel.append(text_vectors)



                # vector_len = len(vector_items)
                # for vector_len in range(max_word_len):

                train_X.append(channel)
                train_y.append(acts_map.index(act_str))

                # print(text_vectors.shape)


            #         print(train_X[0])
            #         print(train_y)
            #        print('len(act_map) : {0}, len(train_X) : {1}, len(train_y) : {2}'.format(len(acts_map), len(train_X), len(train_y)))
        print(acts_map)

    nb_classes = len(train_y)


    print('max word len : ', max_word_len)


    # train_X = train_X.reshape(train_X.shape[0], 1, 21, 50)

    print("Pad sequences (samples x time)")
    # train_X = sequence.pad_sequences(train_X, maxlen=maxlen, dtype='float32')
    train_y = np_utils.to_categorical(train_y, nb_classes)

    '''CNN 을 이용한 text classfication '''
    X_train, X_test, y_train, y_test = train_test_split(train_X, train_y, test_size=0.1)

    X_train = np.asarray(X_train)
    X_test = np.asarray(X_test)
    print(len(X_train), 'train sequences')
    print(len(X_test), 'test sequences')
    print('X_train shape:', X_train.shape)
    print('X_test shape:', X_test.shape)

    # exit()
    model = Sequential()

    model.add(Convolution2D(nb_filters, nb_conv, nb_conv,
                            border_mode='full',
                            input_shape=(1, 21, 50)))
    model.add(Activation('relu'))
    model.add(Convolution2D(nb_filters, nb_conv, nb_conv))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(nb_pool, nb_pool)))
    model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(128))
    model.add(Activation('relu'))
    # model.add(Dropout(0.5))
    model.add(Dropout(0.25))
    model.add(Dense(nb_classes))
    model.add(Activation('softmax'))

    # model.compile(loss='categorical_crossentropy', optimizer='adadelta')
    model.compile(loss='categorical_crossentropy', optimizer='adam')

    model.fit(X_train, y_train, batch_size=batch_size, nb_epoch=nb_epoch, show_accuracy=True, verbose=1,
              validation_data=(X_test, y_test))
	
    #! 스코어 계산
    score = model.evaluate(X_test, y_test, show_accuracy=True, verbose=0)
    print('Test score:', score[0])
    print('Test accuracy:', score[1])