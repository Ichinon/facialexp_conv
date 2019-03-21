import os
import glob
from PIL import Image

#class_path = 'angry'
#class_path = 'disgusted'
#class_path = 'fearful'
#class_path = 'happy'
#class_path = 'sad'
class_path = 'surprised'
input_path = 'E:/Data/RafD/RafD6class/' + class_path
output_path = 'E:/Data/RafD/RafD6class256/' + class_path
output_size = 256

def image_resize(image_file):
    file_name = os.path.basename(image_file)
    im = Image.open(image_file)
    #im = crop_max_square(im)
    im = im.crop((0, 100, 681, 100+681))
    im = im.resize((output_size, output_size))
    im.save(output_path + '/' + file_name + '.png')

files = glob.glob(input_path + "/*")
for image_file in files:
    image_resize(image_file)
