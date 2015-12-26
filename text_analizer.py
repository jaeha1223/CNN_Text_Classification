# -*- coding: utf-8 -*-
'''

dstc2 text analizer

@author: Jeongpil Lee (koreanfeel@gmail.com)
 
'''

import os
import json
import numpy as np
import nltk
from collections import OrderedDict

DATA_FILE = '/Users/lifefeel/Dropbox/음성인식프로젝트/DSTC/DSTC2_DATA/dstc2_traindev/data-train.json'
VECTOR_FILE = '/Users/lifefeel/Downloads/glove/glove.6B.50d.txt'

class TextAnalyzer(object):
    def __init__(self, vector_file):
        self.vocab, self.vectors = self.fromText(vector_file)
        
        vocab_dict = {}
        for i, word in enumerate(self.vocab):
            vocab_dict[word] = i
            
        self.vocab_dict = vocab_dict
        
    def txt2vectors(self, txt):
        txtVectors = OrderedDict()
        tokens = nltk.word_tokenize(txt)
        for token in tokens:
            word = token.lower()
            if word not in self.vocab_dict:
                continue
            else:
                txtVectors[word] = self.vectors[self.vocab_dict[word]]
        return txtVectors

    def fromText(self, file):
        vocabUnicodeSize = 78
        encoding = "utf-8"
        
        with open(file, 'rb') as fin:
            lines = list(fin)
            
            vocab_size = len(lines)
            vector_size = len(lines[0].decode(encoding).strip().split(' ')[1:])
            
            vocab = np.empty(vocab_size, dtype='<U%s' % vocabUnicodeSize)
            vectors = np.empty((vocab_size, vector_size), dtype=np.float)
            for i, line in enumerate(lines):
                line = line.decode(encoding).strip()
                parts = line.split(' ')
                word = parts[0]
                
                vector = np.array(parts[1:], dtype=np.float)
                vocab[i] = word
                vectors[i] = vector
        
        return (vocab, vectors)
 
if __name__ == '__main__':
    transcript = []
    
    ''' json 피알 읽어오기 예제 '''
    with open(DATA_FILE) as f:
        jData = json.load(f)
        
        for jSession in jData:
            jTurns = jSession.get('turns')
            for jTurn in jTurns:
                transcript.append(jTurn.get('system').get('transcript'))
                transcript.append(jTurn.get('user').get('transcript'))

        print(transcript)
    
    ''' txt로부터 vector 가져오기 '''
    ta = TextAnalyzer(VECTOR_FILE)
    print(ta.txt2vectors("I'm a sogang university student."))
    
