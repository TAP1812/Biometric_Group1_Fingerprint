from glob import glob
from crossing_number import *
from normalization import *
import os

import orientation
from frequency import ridge_freq
from gabor_filter import gabor_filter
from segmentation import create_segmented_and_variance_images
from skeletonize import skeletonize

def open_images(db_name, img_name):
    directory = os.path.join('dataset', db_name + '/*')
    images_paths = glob(directory)
    return np.array([cv.imread(img_path,0) for img_path in images_paths if img_name in img_path])


def store_coord(image_name, db_name):
    # Ensure the path is constructed correctly
    DIR_OUTPUT = os.path.join('coord_minu', db_name)
    print(DIR_OUTPUT)
    
    # Create the directory if it doesn't exist
    os.makedirs(DIR_OUTPUT, exist_ok=True)

    images = open_images(db_name, image_name)
    with open(os.path.join(DIR_OUTPUT, image_name + '.txt'), 'a') as f:
        for i, img in enumerate(images):
            print(f"Processing image {i+1}/{len(images)}")
            minu = extract_minu(img)
            f.write(minu.__str__())
            f.write('\n')


def extract_minu(input_img):
    block_size = 16
    normalized_img = normalize(input_img.copy(), float(100), float(100))

    # ROI and normalisation
    (segmented_img, normim, mask) = create_segmented_and_variance_images(normalized_img, block_size, 0.2)

    # orientations
    angles = orientation.calculate_angles(normalized_img, W=block_size, smoth=False)
    orientation_img = orientation.visualize_angles(segmented_img, mask, angles, W=block_size)

    # find the overall frequency of ridges in Wavelet Domain
    freq = ridge_freq(normim, mask, angles, block_size, kernel_size=5, minWaveLength=5, maxWaveLength=15)

    # create gabor filter and do the actual filtering
    gabor_img = gabor_filter(normim, angles, freq)

    # thinning oor skeletonize
    thin_image = skeletonize(gabor_img)

    minutiae_points = []
    biniry_image = np.zeros_like(thin_image)
    biniry_image[thin_image<10] = 1.0
    biniry_image = biniry_image.astype(np.int8)
    (y, x) = thin_image.shape
    for i in range(1, x - 3//2):
        for j in range(1, y - 3//2):
            minutiae, angle = minutiae_at(biniry_image, j, i, 3)
            if minutiae != "none":
                minutiae_points.append([i, j, minutiae, angle])

    minutiae_points = remove_false_minutiaes(minutiae_points)
    return minutiae_points

store_coord('101', 'DB1_B')