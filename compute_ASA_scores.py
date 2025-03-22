import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import os
import BSDS


#superpixels numbers to test
num_superpixels =  [50, 100, 150, 200, 250, 300, 400, 500]

#raeds the image segmentation and returns the array containing the label for each pixel
def segmented_image_to_labels(segmented_img):
    segmented_img = segmented_img.astype(np.uint8)
    h, w, c = segmented_img.shape
    _, labels = np.unique(segmented_img.reshape(-1, c), axis=0, return_inverse=True)
    return labels.reshape(h, w)


#this funcition is used to compute the ASA score for a given segmentation*
def compute_ASA_score(ref_seg, candidate_seg):
    
    print("reference segementation size: ", ref_seg.shape)

    print("condidate segementation size: ", candidate_seg.shape)
    assert ref_seg.size == candidate_seg.size, "differnets size for both segmentations"
    
    N = candidate_seg.size
    
    labels = np.unique(candidate_seg)
    
    Positif_Intersection = 0

    for label in labels:

        label_pixels = ( candidate_seg == label )

        label_ref_pixels = ref_seg[label_pixels]

        nb_ref_pixels_per_label = np.bincount(label_ref_pixels.astype(int))

        Positif_Intersection += np.max(nb_ref_pixels_per_label)

    return Positif_Intersection / N


def get_nspixels(method, n_spixels):
    PATH = f"segmented_images/{method}/{n_spixels}_spixels/"
    images_dict = {}

    if not os.path.exists(PATH):
        print(f"Erreur : Le dossier {PATH} n'existe pas.")
        return images_dict  

    for filename in os.listdir(PATH):
        if filename.endswith(".jpg"):
            img_path = os.path.join(PATH, filename)
            img = mpimg.imread(img_path)
            images_dict[filename] = img

    return images_dict  





def compute_image_ASA(human_segmentation_paths, image):
    """
    Computes the mean ASA score for one image based on its human segmentation paths.

    Parameters:
    -----------
    human_segmentation_paths : list
        List of paths for the human segmentation files.
    image : array-like
        The image corresponding to the segmentation.

    Returns:
    --------
    float
        The mean ASA score for the image.
    """
    N = len(human_segmentation_paths)
    if N == 0:
        return 0

    ASA_sum = 0
    for seg in human_segmentation_paths:
        print(f"ASA score with segmentation {seg}")
        human_segmentation, _ = BSDS.read_segmentation(seg)
        ASA_sum += compute_ASA_score(human_segmentation, image)
    return ASA_sum / N



def compute_ASA_mean_score(num_superpixel):
    """
    Computes the mean Achievable Segmentation Accuracy (ASA) score for a given number of superpixels.

    The ASA score measures the accuracy of the segmentation obtained with the SLIC algorithm 
    by comparing it to human segmentations from the BSDS dataset.

    Parameters:
    -----------
    num_superpixel : int
        Number of superpixels used for SLIC segmentation.

    Returns:
    --------
    float
        The mean ASA score across all images, representing the segmentation accuracy.
    """
    images_with_num_superpixel = get_nspixels("slic", num_superpixel)
    images_number = len(images_with_num_superpixel)

    if images_number == 0:
        print("Error: No superpixel found.")
        return 0

    total_ASA = 0

    for key in images_with_num_superpixel:
        try:
            image_id = int(key.split(".")[0])  # Extract image_id (e.g., "12084.jpg" -> 12084)
        except ValueError:
            print(f"Error: Unable to convert {key} to integer.")
            continue

        human_segmentation_paths = BSDS.find_image_segmetation_path(image_id)
        if not human_segmentation_paths:
            print(f"No segmentation found for image {image_id}.")
            continue
        print(f"computing ASA score for image {image_id}")
        labels = segmented_image_to_labels(images_with_num_superpixel[key])
        ASA_mean = compute_image_ASA(human_segmentation_paths, labels)
        total_ASA += ASA_mean

    return total_ASA / images_number



score = compute_ASA_mean_score(50)
print(score)#0.8863243245556329