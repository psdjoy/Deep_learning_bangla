


# In[5]:


import pandas as pd
import os
import cv2
import numpy as np
import winsound

print("done loading")

# In[2]:




# In[3]:


x = []
folder = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10.oo', '11.aa', '12.e_short', '13.e_long', '14.u_short', '15.u_long', '16.rii', '17.a', '18.oi', '19.o', '20.ou']


for i in folder:
	files = os.listdir(i)
	for j in files:
		print('now reading',j)
		img = cv2.imread('.\\'+i+'\\'+j, 0)
		img = cv2.resize(img, (32, 32))
		x.append(img)

	print('done', i)
	winsound.Beep(2500, 1000)
    
   

print('done', folder)
    
    








# In[6]:





# In[11]:


x = np.array(x)



# In[12]:


np.save('all_x.npy', x)



# In[13]:


print(x.shape)
