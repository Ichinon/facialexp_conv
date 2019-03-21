# -*- coding: utf-8 -*-
"""
Created on Thu Mar 21 13:47:00 2019

@author: iwama
"""
import pickle,numpy as np
from keras.models import load_model
from keras.preprocessing.sequence import pad_sequences

class Text2Emo:
    def __init__(self):
        self.MAX_SEQUENCE_LENGTH = 280
        with open('../../models/txt2emo/tokenizer_cnn_ja.pkl', 'rb') as handle:
            self.tokenizer = pickle.load(handle)
        self.emoClasses  =  ["happy", "sad", "disgust", "angry", "fear", "surprise"]
        self.text2emoModel = load_model('../../models/txt2emo/model_2018-08-28-15_00.h5')

    def detectEmotion(self,text):
        targets = pad_sequences(self.tokenizer.texts_to_sequences(text), maxlen=self.MAX_SEQUENCE_LENGTH)
        emoProb = self.text2emoModel.predict(targets)
        emoProbSoftmax = self.softmax2(emoProb[0])
        emotionList = dict(zip(self.emoClasses, emoProbSoftmax))
        return emotionList

    #ソフトマックス関数を使用すると確率差が小さいため、こちらの関数を使用して確率算出
    def softmax2(self,a):
        c = np.max(a)
        sum_a = np.sum(a)
        y = a / sum_a
        print(y)
        return y
