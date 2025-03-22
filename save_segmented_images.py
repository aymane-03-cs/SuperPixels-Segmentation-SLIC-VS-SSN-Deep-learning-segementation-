import numpy as np
import matplotlib.pyplot as plt
import os
from skimage import io, segmentation, color
import segmentation_methods.deepSSn_segmentation as deepSNN
import segmentation_methods.slic_segmentation as slic
import matplotlib.image as mpimg


BSDS_Path = "BSDS300/images/test/"
OUTPUT_DIR = "segmented_images/"


def compute_and_save_images(nspixels):
    with open("BSDS300/iids_test.txt", 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    out_dir = os.path.join(OUTPUT_DIR, f"slic/{nspixels}_spixels")
    # Le dossier out_dir est supposé exister déjà

    for line in lines:
        img_id = int(line.strip())
        img_path = os.path.join(BSDS_Path, f"{img_id}.jpg")
        img = mpimg.imread(img_path)
        
        # Calculer la segmentation SLIC pour nspixels superpixels
        superpixels = slic.slic_segmentation(img, nspixels)
        segmented_image = slic.segmented_image(img, superpixels)
        
        out_path = os.path.join(out_dir, f"{img_id}.jpg")
        io.imsave(out_path, segmented_image)





#compute_and_save_images(50)
