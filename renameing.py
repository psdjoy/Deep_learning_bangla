import os


file  = os.listdir()
print(file)

jpg = [img for img in file if '.JPG' in img]
print(jpg)

counter = 1
for i in jpg:
	os.rename(i, '%d.JPG'%counter)
	print('renamed', counter)
	counter += 1
