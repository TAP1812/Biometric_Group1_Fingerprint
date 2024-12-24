import cv2 as cv
import numpy as np
import math

def minutiae_at(pixels, i, j, kernel_size):
    if pixels[i][j] == 1:
        if kernel_size == 3:
            cells = [(-1, -1), (-1, 0), (-1, 1),        # p1 p2 p3
                   (0, 1),  (1, 1),  (1, 0),            # p8    p4
                  (1, -1), (0, -1), (-1, -1)]           # p7 p6 p5
        else:
            cells = [(-2, -2), (-2, -1), (-2, 0), (-2, 1), (-2, 2),                 # p1 p2   p3
                   (-1, 2), (0, 2),  (1, 2),  (2, 2), (2, 1), (2, 0),               # p8      p4
                  (2, -1), (2, -2), (1, -2), (0, -2), (-1, -2), (-2, -2)]           # p7 p6   p5

        values = [pixels[i + l][j + k] for k, l in cells]

        crossings = 0
        for k in range(0, len(values)-1):
            crossings += abs(values[k] - values[k + 1])
        crossings //= 2

        if crossings == 1 or crossings == 3:
            angles = []
            for k, (dx, dy) in enumerate(cells[:-1]):
                if values[k] != values[k+1]:
                    angle = math.degrees(math.atan2(dy, dx)) % 360
                    angles.append(angle)

            if angles:
                avg_angle = sum(angles) / len(angles)
            else:
                avg_angle = 0

            if crossings == 1:
                return "ending", avg_angle
            if crossings == 3:
                return "bifurcation", avg_angle

    return "none", None


def remove_false_minutiaes(minutiae_points, min_distance = 10):
    """
    Remove false minutiae based on proximity.
    :param minutiae_points: List of minutiae as (x, y, type).
    :param min_distance: Minimum distance to consider two minutiae as separate.
    :return: Filtered list of minutiae points.
    """
    filtered_minutiae = []
    for i, minutia in enumerate(minutiae_points):
        is_valid = True
        for j, other in enumerate(minutiae_points):
            if i != j:
                distance = np.linalg.norm(np.array(minutia[:2]) - np.array(other[:2]))
                if distance < min_distance:
                    is_valid = False
                    break
        if is_valid:
            filtered_minutiae.append(minutia)
    return filtered_minutiae

def calculate_minutiaes(im, kernel_size=3, filter=False):
    minutiae_points = []
    biniry_image = np.zeros_like(im)
    biniry_image[im<10] = 1.0
    biniry_image = biniry_image.astype(np.int8)

    (y, x) = im.shape
    result = cv.cvtColor(im, cv.COLOR_GRAY2RGB)
    colors = {"ending" : (150, 0, 0), "bifurcation" : (0, 150, 0)}

    for i in range(1, x - kernel_size//2):
        for j in range(1, y - kernel_size//2):
            minutiae, angle = minutiae_at(biniry_image, j, i, kernel_size)
            if minutiae != "none":
                minutiae_points.append([i, j, minutiae, angle])
    
    # Remove false minutiae
    if filter:
        minutiae_points = remove_false_minutiaes(minutiae_points)
    for i, j, minutiae, angle in minutiae_points:
        cv.circle(result, (i,j), radius=2, color=colors[minutiae], thickness=2)
        # Draw orientation line
        length = 10  # Length of the orientation line
        x_end = int(i + length * math.cos(math.radians(angle)))
        y_end = int(j - length * math.sin(math.radians(angle)))
        cv.line(result, (i, j), (x_end, y_end), color=colors[minutiae], thickness=1)

    return  result