# -*- coding: utf-8 -*-
"""
Created on Sun Mar 17 14:02:43 2019

@author: yonet
"""

# モジュールのインポート
import os, tkinter, tkinter.filedialog, tkinter.messagebox,time,itertools
from EmotionDetect import Text2Emo
from PIL import Image


"""
Ichinonさん修正して下さい。
"""
def initializeStarGAN():
    return


"""
Ichinonさん修正して下さい。
"""
def transformImage(baseImage, emotionclass):
    return "./OutputImage/Test.jpg"

def initialize():
    print("===[Start Initilization]====================")
    #StarGAN初期化呼び出し
    initializeStarGAN()

    print("===[End Initilization]======================")
    return


def setBaseImage():
    print("===[Start Base Image Set]====================")

    # ファイル選択ダイアログの表示
    root = tkinter.Tk()
    root.withdraw()
    fTyp = [("","*")]
    iDir = os.path.abspath(os.path.dirname(__file__))
    file = None
    while file is None:
        tkinter.messagebox.showinfo('Deep Communication','ベースとする顔写真を選んで下さい！')
        file = tkinter.filedialog.askopenfilename(filetypes = fTyp,initialdir = iDir)
        print("ベース顔写真:"+file)

    print("===[End Base Image Set]====================")
    return file

def inputText():
    print("===[Start Input Text]======================")
    #ユーザーからメッセージ受け取り
    inputText = None
    while inputText is None:
        print("メッセージを入力して下さい:", end="")
        inputText = input()

    print("===[End Input Text]========================")
    return inputText


def selectEmotion(emotionlist):
    print("===[Start Emotion Selection]====================")
    #感情リストとNLPで計算した確率を提示
    selectedEmotion = None
    while selectedEmotion is None:
        print("感情:確率")
        for key,val in sorted(emotionlist.items(),key=lambda x: -x[1]):
            print(key +":"+ str(int(round(val*100,0))) + "%" )
        print("該当する感情を入力して下さい:",end="")
        tmp = input()
        if tmp is None:
            continue
        elif emotionlist.get(tmp) is None:
            print("リストの中から感情を指定して下さい。")
        else:
            selectedEmotion = tmp
            break

    print("===[End Emotion Selection]====================")
    return selectedEmotion



def presentTransformedImage(transformedImagePath):
    print("===[Start Presentation of Transformed Image]====================")
    #感情変換後の画像表示
    im = Image.open(transformedImagePath)
    im.show()
    print("===[End Presentation of Transformed Image]====================")
    return

def getNextAction():
    nextAction = None
    while nextAction not in ["1","2","3"]:
        print("1.別のテキストを入力しますか?")
        print("2.別のユーザーに切り替えますか?")
        print("3.終了しますか?")
        print("番号で入力して下さい。(1 or 2 or 3):",end="")
        nextAction = input()

    return nextAction

"""
DEEP Communicationのメイン処理

"""
if __name__ == '__main__':
    #Initialization
    initialize()
    text2Emo = Text2Emo()
    userChange = False
    exitProcess = False

    while exitProcess is False:
        #画像ファイル設定
        file = setBaseImage()
        userChange = False

        while userChange is False:
            nextAct = None
            #テキストによる画像変換実施
            presentTransformedImage(transformImage(file, selectEmotion(text2Emo.detectEmotion(inputText()))))
            nextAct = getNextAction()
            if nextAct == "2":
                userChange = True
            elif nextAct == "3":
                exitProcess = True
                break
    print("処理を終了します。お疲れさまでした!!")
