import numpy as np
import cv2 
from sys import argv

def main():
    # Get image to pumpkinize from command line arg
    # TODO: whatever user checks are good practice that I will probably ignore
    img_path = argv[1]
    img = cv2.imread(img_path)
    
    # Convert image to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Segment
    ret, mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # TODO: add morph_ops

    # TODO: put the jack on the lantern
    alpha_mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2RGBA)
    alpha_mask[np.all(alpha_mask == [255, 255, 255, 255], axis=2)] = [0, 0, 0, 0]

    # Save final image
    img_path_split = img_path.split('.')
    cv2.imwrite(img_path_split[0] + '_pumpkin.png', mask)

main()
