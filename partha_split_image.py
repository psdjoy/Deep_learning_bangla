


import image_slicer
import os






img = 'u4.jpg'  #name has to be changed
a = image_slicer.slice(img, 256)
image_slicer.save_tiles(a)

