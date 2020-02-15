import requests
import re
from os import sys

r = requests.get('https://www.cwb.gov.tw/Data/js/obs_img/Observe_radar.js')

if r.status_code != 200:
	print("Failed to load Page.")
	exit(0)
print("Loaded page successfully.")

cnt = 0

for s in re.finditer('CV1_TW_3600_2020', r.text):
	file = r.text[s.start():s.start()+28]
	img = requests.get('https://www.cwb.gov.tw/Data/radar/' + file)
	with open('./input_file/' + file, 'wb') as f:
		f.write(img.content)

	sys.stdout.flush()
	sys.stdout.write('\r[' + str(int(cnt * 100 / 90)) + '%]')
	sys.stdout.flush()
	cnt = cnt + 1
sys.stdout.flush()
print('\nDone.')
