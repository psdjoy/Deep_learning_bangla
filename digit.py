import pandas as pd
import numpy as np
import os
from shutil import copy

data = pd.read_csv('training-a.csv')

digit = data['digit']
name = data['filename']

print(len(digit))
print(len(name))

try:
	for i in range(0,10):
		os.mkdir(str(i))
except:
	pass


for i in range(0,len(digit)):
	d = digit[i]
	f_name = name[i]
	copy('./training-a/'+f_name, './'+str(d))
	print('coppied', f_name)