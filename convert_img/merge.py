#takes average of every pixel for every 6 image in an hour
from PIL import Image
from os import listdir, sys
from datetime import datetime
import numpy as np
import re

t1 = datetime.now()
default_re = ".*.png"
available_for_merge = ['']
l=list(listdir(".\\output_file"))
l2=list(listdir(".\\merged_file"))
for file in l:
	if file[:10] in available_for_merge or (file[:10] + '.png') in l2:
		continue
	if len(list(filter(re.compile(file[:10]).match,l))) == 6:
		available_for_merge.append(file[:10])
del available_for_merge[0]

if len(available_for_merge) == 0:
	print("Nothing to merge.")
	exit(0)
	
while 1:
	print("Available regular expressions:")
	for file in available_for_merge:
		print(file)

	default_re = input("\nenter regular expression\n")
	r = re.compile(default_re)
	l=list(listdir(".\\output_file"))
	newlist = list(filter(r.match,l)) # Read Note
	print("file match:")

	if not default_re in available_for_merge:
		print("Not available for merge. try again.\n")
		continue

	for i in newlist:
		print(i)
	confirm = input("comfirm search result?(y/n)")

	if(confirm == 'y'):
		break
	elif(confirm =='n'):
		print("current expression: "+default_re)
		default_re = input("please input new regular expression\n")
	else:
		print("invalid input")

im = [Image.open('./output_file/'+newlist[i]).load() for i in range(6)]
w, h = 3600, 3600
arr = np.zeros([h, w], dtype = np.uint8)
for x in range(h):
	for y in range(w):
		avg = 0
		for i in range(6):
			avg += im[i][y, x]
		arr[x, y] = avg / 6
	sys.stdout.flush()
	sys.stdout.write("\r["+str(int(x/(h-1)*100))+"%]")
	sys.stdout.flush()
newimg = Image.fromarray(arr)
newimg.save('./merged_file/'+default_re+'.png')
print('\n' + datetime.now() - t1)