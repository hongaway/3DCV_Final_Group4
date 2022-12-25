from PIL import Image
import numpy as np
mask = Image.open('./000000-label.png')
mask = np.array(mask)
mask[ mask>0] = 255
mask = Image.fromarray(mask)
bg = Image.open('/home/ai2lab/Desktop/台達計畫/T6D_implement/T6D-direct/YCB_Video_Dataset/backgrounds/nvidia/nvidia1/000000-color.png')
ob = Image.open('./000000-color.png')
im_np = np.array(mask)
im2_np = np.array(bg)
ob_np = np.array(ob)
im = Image.composite(ob, bg, mask)
im.show()