"""
Test

"""
from glob import glob
from crossing_number import remove_false_minutiaes, minutiae_at
from normalization import *
import os
import cv2 as cv
from orientation import *
from frequency import ridge_freq
from gabor_filter import gabor_filter
from segmentation import create_segmented_and_variance_images
from skeletonize import skeletonize
import itertools
from scipy.spatial.distance import pdist, squareform
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report

def open_images(db_name, img_name, num_images=80):
    directory = os.path.join('dataset', db_name + '/*')
    images_paths = glob(directory)
    images_paths = images_paths[:num_images]
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
            minu = create_grid_histogram(minu, (388, 374))
            f.write(minu.__str__())
            f.write('\n')


def extract_minu(input_img):
    block_size = 16
    normalized_img = normalize(input_img.copy(), float(100), float(100))

    # ROI and normalisation
    (segmented_img, normim, mask) = create_segmented_and_variance_images(normalized_img, block_size, 0.2)

    # orientations
    angles = calculate_angles(normalized_img, W=block_size, smoth=False)
    orientation_img = visualize_angles(segmented_img, mask, angles, W=block_size)

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

def minu_normalization_vector(minutiae_points):
    """
    Normalizes the minutiae points to be in the range [0, 1]
    :param minutiae_points: List of minutiae points as (x, y, type, angle).
    :return: Normalized list of minutiae points.
    """
    VERTICAL_SIZE = 374
    HORIZONTAL_SIZE = 388

    return [[minutia[0] / HORIZONTAL_SIZE, minutia[1] / VERTICAL_SIZE, minutia[2], minutia[3]] for minutia in minutiae_points]

def geometric_features(minutiae_points):
    """
    Extracts the geometric features from the minutiae points.
    :param minutiae_points: List of minutiae points as (x, y, type, angle).
    :return: Pairwise distances between minutiae points, and the number of bifurcations and endings.
    """
    bifurcations = 0
    endings = 0
    for minutia in minutiae_points:
        if minutia[2] == "bifurcation":
            bifurcations += 1
        if minutia[2] == "ending":
            endings += 1
    
    coords = np.array([[minutia[0], minutia[1]] for minutia in minutiae_points])
    angles = np.array([minutia[3] for minutia in minutiae_points])

    distances = pdist(coords)
    distance_matrix = squareform(distances)

    angle_matrix = np.abs(angles[:, None] - angles)
    angle_matrix[angle_matrix > 180] = 360 - angle_matrix[angle_matrix > 180]

    feature_vector = np.concatenate([distance_matrix.flatten(), angle_matrix.flatten()])

    return feature_vector

def create_grid_histogram(minutiae_points, image_shape, grid_size=(10, 10)):
    rows, cols = image_shape
    grid_rows, grid_cols = grid_size
    
    # Divide the image into grid cells
    cell_height = rows // grid_rows
    cell_width = cols // grid_cols
    
    # Initialize histogram
    histogram = np.zeros((grid_rows, grid_cols, 2))  # Separate bins for endings and bifurcations

    for x, y, minutiae_type, angle in minutiae_points:
        row_idx = min(y // cell_height, grid_rows - 1)
        col_idx = min(x // cell_width, grid_cols - 1)
        if minutiae_type == "ending":
            histogram[row_idx, col_idx, 0] += 1
        elif minutiae_type == "bifurcation":
            histogram[row_idx, col_idx, 1] += 1

    # Flatten the histogram
    return histogram.flatten().tolist()

def get_features_vector(image_name, db_name):
    images = open_images(db_name, image_name)
    print(len(images))
    features = []
    for i, img in enumerate(images):
        print(f"Processing image {i+1}/{len(images)} of image {image_name} of database {db_name}")
        minu = extract_minu(img)
        features.append(create_grid_histogram(minu, (388, 374)))
    return features


labels = []
for db_name in ['DB1_B', 'DB2_B', 'DB3_B', 'DB4_B']:
    for i in range(101, 111):
        for _ in range(8):
            labels.append(f'{db_name}_{i}')

y = np.array(labels)



X = []
for db_name in tqdm(['DB1_B', 'DB2_B', 'DB3_B', 'DB4_B']):
    for img_name in range(101, 111):
        temp = get_features_vector(str(img_name), db_name)
        for vector in temp:
            X.append(vector)
X = np.array(X)




X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.125, random_state=42)

knn = KNeighborsClassifier(n_neighbors=4)  # Specify the number of neighbors (k)
knn.fit(X_train, y_train)

y_pred = knn.predict(X_test)

# Step 4: Evaluate the Model
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)
print("Classification Report:\n", classification_report(y_test, y_pred))




