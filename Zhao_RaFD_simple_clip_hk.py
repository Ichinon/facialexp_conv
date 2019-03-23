import cv2
import os
import glob

size = (256,256)
data_root_RaFD = 'E:/Data/RafD/RafD8class/'
save_root_path = 'E:/Data/RafD/RafD8class256/'
os.chdir(data_root_RaFD)
emos = glob.glob('*')

for emo in emos:
    data_path = data_root_RaFD + emo
    save_path = save_root_path + emo
    
    os.chdir(data_path)
    
    pics = glob.glob('*.jpg') 
    pics = [s for s in pics if not "Rafd000" in s]
    pics = [s for s in pics if not "Rafd180" in s]
    
    profile_list = []
 
    for pic in pics:
        pic_name = pic[:-4]
#         print(pic)
#         print(data_path + "\\" + pic)
        src = cv2.imread(data_path + "\\" + pic)

        cv2.imwrite(save_path + "\\" + pic_name + ".png", cv2.resize(src[130 : 130 + 600,  20 : 20 + 600], size))
