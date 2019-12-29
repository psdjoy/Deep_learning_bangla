#at first install image_slicer in windowsby writting "pip install image_slicer"


import image_slicer
import os

img = 'name.jpg'  #name has to be changed
a = image_slicer.slice(img, 256)
image_slicer.save_tiles(a)

