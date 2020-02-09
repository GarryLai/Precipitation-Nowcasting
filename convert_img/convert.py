from PIL import Image
import numpy as np
import switch


def get_img(file):
	try:
		im = Image.open(file).convert("RGB")
	except FileNotFoundError:
		print("File not found.")
		exit(0)

	return im

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

def get_dbz(arr):
 	#arr must be 1d nparray with a size of 3
 	dbz = switch.color_dict.get(arr.tobytes())
 	if dbz:
 		return dbz
 	else:
 		return -1
		
def grayify(filename, im, w, h):
	pix = np.array(im)
	arr = np.zeros([h, w], dtype=np.uint8)
	l = 0
	r = 0
	for x in range(h):
		for y in range(w):
			t = get_dbz(pix[x][y])
			if t == -1:
				if get_dbz(pix[x, min(w-1, y+10)]) != -1:
					arr[x, y:min(w-1, y+10)+1] = 255*(get_dbz(pix[x-1, min(w-1, y+10)])/65)
				elif get_dbz(pix[min(h-1, x+10), y]) != -1:
					arr[x:min(h-1, w+10)+1, y] = 255*(get_dbz(pix[min(h-1, x+10), y])/65)
				else:
					arr[x, y]=0	
			else:
				arr[x, y] =  255 * (t / 65)
	newimg = Image.fromarray(arr)
	# newimg.show()
	newimg.save(filename+'_greyed.png')

if __name__=='__main__':
	file = input("img name (without .png): ")
	im = get_img(file+'.png')
	w, h = im.size
	l, u, r, d = [0, 0, w, h] #get_boundary(w, h)

	cropped_img = im.crop((l, u, r, d))
	grayify(file, cropped_img, r - l, d - u)
	print('Done.')

