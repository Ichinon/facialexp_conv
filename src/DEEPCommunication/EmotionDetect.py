# -*- coding: utf-8 -*-
"""
Created on Thu Mar 21 13:47:00 2019

@author: iwama
"""
import fasttext as ft
from janome.tokenizer import Tokenizer
import os

class Text2Emo:
    def __init__(self):
        self.tokenizer = Tokenizer()
        self.emoClasses  =  {"__label__0":"happy","__label__1":"sad","__label__2":"disgust","__label__3":"angry","__label__4":"fear","__label__5":"surprised"}
        self.text2emoModel = ft.load_model(os.path.join(os.path.dirname(__file__), "fasttext_model_30.bin"))

    def detectEmotion(self,text):
        tokens = self.tokenizer.tokenize(text)
        wakatis = []
        wakati_list=[]
        for token in tokens:
            wakati_list.append(token.surface)
            wakati=" ".join(wakati_list)
        wakatis.append(wakati)

        estimate = self.text2emoModel.predict_proba(wakatis,k=6)
        emoProbList = dict(estimate[0])
        emotionList={}
        for key in emoProbList:
          emotionList[self.emoClasses[key]] = emoProbList[key]
        return emotionList
