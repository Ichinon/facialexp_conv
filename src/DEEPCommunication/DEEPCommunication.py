# -*- coding: utf-8 -*-
"""
Created on Sun Mar 17 14:02:43 2019

@author: yonet
"""

# モジュールのインポート
import os, tkinter, tkinter.filedialog, tkinter.messagebox,time,itertools
from EmotionDetect import Text2Emo
from pixyz.distributions import Deterministic, DataDistribution
from PIL import Image
import shutil
import sys
import torch
import numpy as np
import cv2

## StarGAN用のパラメータ
# stargan.pyのインポート
sys.path.append(os.path.join(os.path.dirname(__file__), '../stg'))
from stargan_pixyz import StarGAN, Generator

# 学習済みモデルディレクトリ
ganModelDir = os.path.join(os.path.dirname(__file__), '../../models/emo2img256')
# 推論用入力画像配置ディレクトリ
inpImageDir = os.path.join(os.path.dirname(__file__), '../../inp/production')
# 出力ファイルパス
resImagePath = os.path.join(os.path.dirname(__file__), '../../res/result.gif')
# 感情クラス数
c_dim = 7
# 生成画像サイズ
image_size = 256
# gif出力モード
mode = 'test_mv'
# modelのiteration番号
resume_iters = 200000

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# Emotional class dist.
c = DataDistribution(["c"]).to(device)
# Data dist.
p_data = DataDistribution(["x"]).to(device)
# generator
G = Generator(conv_dim=64, c_dim=c_dim, repeat_num=6).to(device)

# CV2 顔検出用モデルファイル Hironobu-Kawaguchi
face_cascade_path = os.path.join(os.path.dirname(__file__),'../../models/cv2/haarcascade_frontalface_default.xml')

### OpenCVで顔検出し、inputフォルダに格納   Hironobu-Kawaguchi 2019.3.24
def imread(filename, flags=cv2.IMREAD_COLOR, dtype=np.uint8):   ### 日本語のパスに対応用
    try:
        n = np.fromfile(filename, dtype)
        img = cv2.imdecode(n, flags)
        return img
    except Exception as e:
        print(e)
        return None
    
def imwrite(filename, img, params=None):
    try:
        ext = os.path.splitext(filename)[1]
        result, n = cv2.imencode(ext, img, params)

        if result:
            with open(filename, mode='w+b') as f:
                n.tofile(f)
            return True
        else:
            return False
    except Exception as e:
        print(e)
        return False    

def save_faceImage(baseImage, crop_size=256, margin_c=0.5):
    size = (crop_size, crop_size)
    #print(baseImage)
    #src = cv2.imread(baseImage)
    src = imread(baseImage)     ### 日本語のパスに対応用
    #print(type(src))
    basename = os.path.basename(baseImage)
    pic_name = basename[:-4]
    face_cascade = cv2.CascadeClassifier(face_cascade_path)
    src_gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(src_gray)
    print("detect", faces.shape[0], "faces", basename)

    for i, face_detect in enumerate(faces):
        leftmargin = face_detect[0] / face_detect[2]
        rightmargin = (src.shape[1] - face_detect[0] - face_detect[2]) / face_detect[2]
        upmargin = face_detect[1] / face_detect[3]
        downmargin  = (src.shape[0] - face_detect[1] - face_detect[3]) / face_detect[3]
        margin = min(margin_c, leftmargin, rightmargin, upmargin, downmargin)
        #print(src.shape, margin)
        x = int(face_detect[0] - face_detect[2] * margin)
        y = int(face_detect[1] - face_detect[3] * margin)
        w = int(face_detect[2] * (1 + margin*2))
        h = int(face_detect[3] * (1 + margin*2))
        face = src[y: y+h, x: x+w]

        # outfile_name = pic_name + str(i) + ".png"
        # inpImage = os.path.join(inpImageDir, 'neu', outfile_name)
        #print("face [x, y, w, h] =", face_detect, basename, "->" , outfile_name)

        inpImage = os.path.join(inpImageDir, 'neu', 'inp.png')
        imwrite(inpImage, cv2.resize(face, size))
    return


def transformImage(baseImage, emotionclass):

    # 入力画像を所定のサイズで切り出しし所定の場所に保存
    save_faceImage(baseImage, crop_size=256, margin_c=0.5)

    # label生成
    # 0:angry 1:disgusted 2:fearful 3:happy 4:neutral 5:sad 6:surprised
    if emotionclass == "happy":
        c_trg = torch.Tensor([3])
    elif emotionclass == "sad":
        c_trg = torch.Tensor([5])
    elif emotionclass == "disgust":
        c_trg = torch.Tensor([1])
    elif emotionclass == "angry":
        c_trg = torch.Tensor([0])
    elif emotionclass == "fear":
        c_trg = torch.Tensor([2])
    elif emotionclass == "surprised":
        c_trg = torch.Tensor([6])
    else:
        print('不適切な感情クラスが入力されため、画像生成しません.')
        return None

    # 画像生成
    emo2Img.test(resImagePath, c_trg, torch.Tensor([4]), c_dim)

    return resImagePath


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
    import subprocess
    cmd = 'start' +  ' ' + transformedImagePath
    subprocess.call(cmd, shell=True)
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
    print("===[Start Initilization]====================")
    emo2Img = StarGAN(p_data, G, c, inpImageDir, mode, c_dim=c_dim, image_size=image_size, resume_iters=resume_iters, model_save_dir=ganModelDir)
    text2Emo = Text2Emo()
    print("===[End Initilization]======================")
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
