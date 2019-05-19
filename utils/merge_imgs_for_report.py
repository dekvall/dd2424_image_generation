import cv2
import glob
import matplotlib.pyplot as plt
import numpy as np
from collections import OrderedDict, defaultdict

imgs = OrderedDict()
for imgname in sorted(glob.glob('../results/eval/*.jpg')):
	fn = imgname.split('/')[-1][:-4]
	imgs[fn] = cv2.resize(cv2.imread(imgname), (128,128))
for imgname in sorted(glob.glob('../results/eval/*.png')):
	fn = imgname.split('/')[-1][:-4]
	imgs[fn] = cv2.resize(cv2.imread(imgname), (128,128))

res = defaultdict(list)
for name, img in imgs.iteritems():
	res[name[:4]].append(img)

girl = np.hstack(res['girl'])
suit = np.hstack(res['jack'])
jack = np.hstack(res['suit'])

cv2.imwrite('../results/eval/merge_girl.png', girl)
cv2.imwrite('../results/eval/merge_suit.png', suit)
cv2.imwrite('../results/eval/merge_jack.png', jack)
cv2.imshow('gg',girl)
cv2.waitKey()
cv2.imshow('gg',jack)
cv2.waitKey()
cv2.imshow('gg',suit)
cv2.waitKey()




