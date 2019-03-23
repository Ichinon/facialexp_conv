import cv2
import os
import glob

size = (256,256)
data_root_RaFD = 'C:/Users/PXK13/hk/StarGAN/input/matsuo/'
save_root_path = 'C:/Users/PXK13/hk/StarGAN/input/matsuo256/'
face_cascade_path = "haarcascade_frontalface_default.xml"   # 同じフォルダに入れてみた
face_cascade = cv2.CascadeClassifier(face_cascade_path)
margin_c = 0.5     # 顔切り抜きの拡大比率(固定値)

os.chdir(data_root_RaFD)
emos = glob.glob('*')

pics = glob.glob('*') 

for pic in pics:
    pic_name = pic[:-4]
#         print(pic)
    src = cv2.imread(data_root_RaFD + "\\" + pic)
    #print(src.shape)

    src_gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(src_gray)
    print("detect", faces.shape[0], "faces", pic)

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

        outfile_name = pic_name + str(i) + ".png"
        print("face [x, y, w, h] =", face_detect, pic, "->" , outfile_name)
        cv2.imwrite(save_root_path + "\\" + outfile_name, cv2.resize(face, size))
