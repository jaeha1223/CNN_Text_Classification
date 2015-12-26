# -*- coding: utf-8 -*-
'''

dstc2 text analizer

@author: Jeongpil Lee (koreanfeel@gmail.com)

'''

import os
import json
import numpy as np
from sklearn import svm
import text_analizer as ta
from sklearn.cross_validation import train_test_split

DATA_FILE = '/home/public_data/dstc2/data-train.json'
VECTOR_FILE = '/home/public_data/glove/glove.6B.50d.txt'

if __name__ == '__main__':

    acts_map = []
    train_X = []
    train_y = []

    ''' txt로부터 vector 가져오기 '''
    ta = ta.TextAnalyzer(VECTOR_FILE)
#     print(ta.txt2vectors("I'm a sogang university student."))

    ''' json 파일에서 필요한 정보 읽어오기 '''
    with open(DATA_FILE) as f:
        jData = json.load(f)

        for jSession in jData:
            jTurns = jSession.get('turns')
            for jTurn in jTurns:
                text = jTurn.get('user').get('transcript')

                dialog_acts = jTurn.get('user').get('dialog-acts')

                acts = [] # 한 턴에 act가 여러개인 경우를 처리하기 위한 리스트

                for dialog_act in dialog_acts:
                    if dialog_act['act'] not in acts:
                        acts.append(dialog_act['act'])

                acts.sort()
                act_str = '-'.join(acts) #act가 여러개 일 경우 - 으로 연결하여 하나의 조합의로 처리하기 위함

                if act_str not in acts_map:
                    acts_map.append(act_str)

                vector_items = ta.txt2vectors(text) # 이 부분에서 text를 vector로 변환함

                ''' average vector 를 계산 '''
                vectors_sum = 0
                for word in vector_items:
                    vectors_sum = np.add(vectors_sum, vector_items[word])

                if len(vector_items) > 0:
                    avg_vector = np.nan_to_num(vectors_sum / len(vector_items))
                else:
                    avg_vector = np.zeros(50, dtype=float)

                train_X.append(avg_vector)
                train_y.append(acts_map.index(act_str))

        '''svm 을 이용한 classfication '''
        X_train, X_test, y_train, y_test = train_test_split(train_X, train_y, test_size=0.1)
        clf = svm.SVC()
        clf.fit(X_train, y_train)

        predict = clf.predict(X_test)

        total = len(X_test)
        correct = 0
        for i in range(len(X_test)):
            if predict[i] == y_test[i]:
                correct += 1

        print('TEST : {0} / {1}, {2:.2f}%'.format(correct, total, correct / total * 100))