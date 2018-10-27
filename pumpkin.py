import numpy as np
import cv2 
from sys import argv

def morph_ops(mask):
    little_kernel = np.ones((3,3), np.uint8)
    kernel = np.ones((5,5), np.uint8)
    big_kernel = np.ones((9,9), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_DILATE, little_kernel, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, big_kernel, iterations=5)
    smoothed = cv2.blur(mask, (19, 19))
    ret, smoothed = cv2.threshold(smoothed, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    return smoothed


def main():
    # Get image to pumpkinize from command line arg
    # TODO: whatever user checks are good practice that I will probably ignore
    img_path = argv[1]

    pum = cv2.imread('pumpkin.jpg')
    img = cv2.imread(img_path)
    img = cv2.resize(img, (pum.shape[1], pum.shape[0]))

    print(pum.shape)
    rows, cols, ch = img.shape
    roi = pum[0:rows, 0:cols]
    
    # Convert image to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Segment
    ret, mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY)
    mask = morph_ops(mask)
    mask_inv = cv2.bitwise_not(mask)

    # TODO: add morph_ops

    # Load pumpkin
    light = cv2.imread('light.jpg')
    light_size = cv2.resize(light, pum.shape[0:2])

    print(img.shape)
    print(mask_inv.shape)
    print(roi.shape)

    pum_bg = cv2.bitwise_and(roi, roi, mask = np.uint8(mask_inv))
    light_bg = cv2.bitwise_and(img, img, mask = np.uint8(mask))
    both = cv2.add(pum_bg, light_bg)
    pum[0:rows, 0:cols] = both

    # Save final image
    img_path_split = img_path.split('.')
    cv2.imwrite(img_path_split[0] + '_pumpkin.png', both)

main()
