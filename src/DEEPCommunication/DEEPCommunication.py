# -*- coding: utf-8 -*-
"""
Created on Sun Mar 17 14:02:43 2019

@author: yonet
"""

# モジュールのインポート
import os, tkinter, tkinter.filedialog, tkinter.messagebox
from PIL import Image
import shutil
import sys
import torch

## StarGAN用のパラメータ
# stargan.pyのインポート
sys.path.append(os.path.join(os.path.dirname(__file__), '../stg'))
import stargan

# 学習済みモデルディレクトリ
ganModelDir = os.path.join(os.path.dirname(__file__), '../../models/emo2img256')
# 推論用入力画像配置ディレクトリ
inpImageDir = os.path.join(os.path.dirname(__file__), '../../inp/production')
# 出力ファイルパス
resImagePath = os.path.join(os.path.dirname(__file__), '../../res/result.gif')
# 感情クラス数
c_dim = 7
# ジェネレータのインスタンス
G = stargan.Generator(conv_dim=64, c_dim=7, repeat_num=6)
G.to(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))

"""
岩間さん修正して下さい
"""
def initializeEmotionDetect():
    return

"""
Ichinonさん修正して下さい。
"""
def initializeStarGAN():
    resume_iters = 200000
    G_path = os.path.join(ganModelDir, '{}-G.ckpt'.format(resume_iters))
    G.load_state_dict(torch.load(G_path, map_location=lambda storage, loc: storage))
    return

"""
岩間さん修正して下さい
"""
def detectEmotion(text):
    return {"angry":0.5, "sad":0.3, "happy":0.2}

"""
Ichinonさん修正して下さい。
""" 
def transformImage(baseImage, emotionclass):
    # baseImageを所定のフォルダにコピー
    basename = os.path.basename(baseImage)
    inpImage = os.path.join(inpImageDir, 'neu', basename)
    shutil.copyfile(baseImage, inpImage)

    # 画像生成
    stargan.test_mv(G, inpImageDir, resImagePath, torch.Tensor([2]), c_dim)

    return resImagePath

def initialize():
    print("===[Start Initilization]====================")
    6
    #NLP初期化呼び出し
    initializeEmotionDetect()
    
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
        for key,val in emotionlist.items():
            print(key +":"+ str(val) )
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
    import subprocess
    cmd = 'start' +  ' ' + transformedImagePath
    subprocess.call(cmd, shell=True)
    # im = Image.open(transformedImagePath)    
    # im.show()

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
    
    userChange = False
    exitProcess = False
    
    while exitProcess is False:
        #画像ファイル設定
        file = setBaseImage()
        userChange = False
        
        while userChange is False:
            nextAct = None
            #テキストによる画像変換実施
            presentTransformedImage(transformImage(file, selectEmotion(detectEmotion(inputText()))))
            nextAct = getNextAction()
            if nextAct == "2":
                userChange = True
            elif nextAct == "3":
                exitProcess = True
                break
    print("処理を終了します。お疲れさまでした!!")
    
    
