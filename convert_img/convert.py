from PIL import Image
from datetime import datetime
from os import listdir
import numpy as np
import re
import switch

color_dict = switch.color_dict

def get_files():
	default_re = ".*.png"
	while 1:
		r = re.compile(default_re)
		l=list(listdir(".\\input_file"))
		newlist = list(filter(r.match,l)) # Read Note
		print("file match:")
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
	return newlist

def get_boundary(w, h):
	b = list(map(int, input("crop boundry(l, u, r, d)(separated by space): ").split()))

	if len(b) != 4:
		print("Input format incorrect. (l, r, u, d)")
		exit(0)

	l, u, r, d = b
	if not (l>=0 and l<r and u>=0 and u<d and r>l and r<w and d>u and d<h):
		print("Input out of range.")
		exit(0)

	return b

def get_dbz(tup):
 	#arr must be 1d nparray with a size of 3
 	dbz = color_dict.get(tup)
 	if dbz:
 		return dbz
 	else:
 		return -1
		
def greyify(filename, im, w, h):
	pix = im.load()
	arr = np.zeros([h, w], dtype = np.uint8)
	for x in range(h):
		for y in range(w):
			t = get_dbz(pix[y, x])
			if t == -1:
				if get_dbz(pix[min(w-1, y+10), x]) != -1:
				 	arr[x, y:min(w-1, y+10)+1] = 255*(1 - get_dbz(pix[min(w-1, y+10), x]))
				elif get_dbz(pix[y, min(h-1, x+10)]) != -1:
				  	arr[x:min(h-1, x+10)+1, y] = 255*(1 - get_dbz(pix[y, min(h-1, x+10)]))
				else:
					arr[x, y]= 0	
			else:
				arr[x, y] =  255  * (1 - t)
	newimg = Image.fromarray(arr)
	newimg.show()
	newimg.save(".\\output_file\\"+filename[:-4]+'_greyed.png')

if __name__=='__main__':
	for file in get_files():
		t1 = datetime.now()
		im = Image.open(".\\input_file\\"+file)
		w, h = im.size
		greyify(file, im, w, h)
		t2 = datetime.now()
		print('Done.')
		print(str(t2-t1)+'\n')

