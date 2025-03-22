import save_segmented_images as seg_imgs
import compute_ASA_scores


num_superpixels =  [50, 100, 150, 200, 250, 300, 400, 500]

SLIC_scores = {}
dSSN_scores = {}

if __name__ == 'main':

    for ns in num_superpixels:
        seg_imgs.compute_and_save_images(ns)

    for ns in num_superpixels:
        score = compute_ASA_scores.compute_ASA_mean_score(ns)
        SLIC_scores[f"ns"] = score