import numpy as np
import cv2 
from skimage.measure import compare_psnr
from skimage.measure import compare_ssim

### TEST USING BSD - 50 EPOCHS 
# img1 = cv2.imread('test/test_real_0004.png',0)
# img2 = cv2.imread('test/test_gen_0004.png',0)

# print(compare_psnr(img1,img2))
# print(compare_ssim(img1,img2, data_range=img2.max()-img2.min()))

# 4 images = 25.1975

psnr = 0
ssim = 0
numImg = 16
for i in range(0,numImg):
	# print(i)
	img1 = cv2.imread('test/paper/75/color/previous/scenario_vgg/16x16/test_real_{:04d}.png'.format(i+1),1) 	# set jadi 1 kalo rgb
	img2 = cv2.imread('test/paper/75/color/previous/scenario_vgg/16x16/test_gen_{:04d}.png'.format(i+1),1) 	# set jadi 1 kalo rgb
	psnr = psnr + compare_psnr(img1,img2)
	ssim = ssim + compare_ssim(img1,img2, data_range=img2.max()-img2.min(), multichannel=True) # kalo rgb, set multichannel = TRUE

print(psnr / numImg)
print(ssim / numImg)
