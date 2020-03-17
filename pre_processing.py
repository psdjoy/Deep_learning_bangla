


# In[5]:


import pandas as pd
import os
import cv2
import numpy as np
import csv
import winsound

print("done loading")

# In[2]:




# In[3]:


x = []
y = []


folder = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14', '15', '16', '17', '18', '19', '20']


for i in folder:
	files = os.listdir(i)
	for j in files:
		print('now reading',j)
		img = cv2.imread('.\\'+i+'\\'+j, 0)
		img = cv2.resize(img, (64, 64))
		(thresh, img) = cv2.threshold(img, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
		x.append(img)
		y.append(int(i))

	print('done', i)
	winsound.Beep(1500, 500)
    
   

print('done', folder)
    
    
y = np.array([y])
y = np.transpose(y)
np.savetxt('y.csv', y, delimiter="")







# In[6]:





# In[11]:


x = np.array(x)




# In[12]:


np.save('all_x.npy', x)



# In[13]:


print(x.shape)
print(len(y))
