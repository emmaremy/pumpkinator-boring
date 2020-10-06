import numpy as np
import cv2 
from sys import argv
import imageio

# Command line arg: file path to image to pumpkinize

def morph_ops(mask):
    little_kernel = np.ones((2,2), np.uint8)
    kernel = np.ones((3,3), np.uint8)
    big_kernel = np.ones((5,5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_DILATE, little_kernel, iterations=1)
    #mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=5)
    smoothed = cv2.blur(mask, (1, 1))
    ret, smoothed = cv2.threshold(smoothed, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    return smoothed


def main():
    # Get image to pumpkinize from command line arg
    # TODO: whatever user checks are good practice that I will probably ignore
    img_path = argv[1]

    # Load pumpkin
    pum = cv2.imread('pumpkin.jpg')

    # Load cutout image
    img = cv2.imread(img_path)

    # Resize everything
    print(pum.shape)
    cols, rows, ch = img.shape
    print(img.shape)
    if rows > cols:
        img_psize = cv2.resize(img, (280, int(280*float(cols)/rows)))
    else:
        img_psize = cv2.resize(img, (int(200*float(rows)/cols), 200))
    cols, rows, ch = img_psize.shape
    print(img_psize.shape)
    roi = pum[135:cols+135, 166:rows+166]
    print(roi.shape)

    # Convert image to grayscale
    gray = cv2.cvtColor(img_psize, cv2.COLOR_BGR2GRAY)

    # Segment
    ret, mask = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    mask = morph_ops(mask)
    mask_inv = cv2.bitwise_not(mask)

    # Load light
    light = cv2.imread('light.jpg')
    light = cv2.resize(light, (pum.shape[1], pum.shape[0]))
    light = light[135:cols+135, 166:rows+166]
    light_darker = light - 20

    # Combine everything
    pum_bg = cv2.bitwise_and(roi, roi, mask = np.uint8(mask))
    light_bg = cv2.bitwise_and(light, light, mask = np.uint8(mask_inv))
    light_bg_darker = cv2.bitwise_and(light_darker, light_darker, mask = np.uint8(mask_inv))
    both = cv2.add(pum_bg, light_bg)
    both_darker = cv2.add(pum_bg, light_bg_darker)
    pum[135:cols+135, 166:rows+166] = both
    pum_darker = pum.copy()
    pum_darker[135:cols+135, 166:rows+166] = both_darker
    pum_g = cv2.cvtColor(pum, cv2.COLOR_RGB2BGR)
    pum_darker_g = cv2.cvtColor(pum_darker, cv2.COLOR_RGB2BGR)
    images = [pum_g, pum_darker_g]

    # Save final image
    img_path_split = img_path.split('.')
    cv2.imwrite(img_path_split[0] + '_pumpkin.png', pum)
    imageio.mimsave(img_path_split[0] + '_pumpkin.gif', images)

main()
