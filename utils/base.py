import cv2
import numpy as np
from scipy import ndimage
from matplotlib.colors import Normalize
from sklearn.neighbors import KernelDensity
import matplotlib.pyplot as plt
import copy


def filter_objects_size(mask, size_th, dir):
    """
    Filter objects in a binary mask by size
    :param mask: A binary mask to filter
    :param size_th: The size threshold used to filter (objects GREATER than the threshold will be kept)
    :return: A binary mask containing only objects greater than the specified threshold
    """
    _, output, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    sizes = stats[1:, -1]
    if dir == "greater":
        idx = (np.where(sizes > size_th)[0] + 1).tolist()
    if dir == "smaller":
        idx = (np.where(sizes < size_th)[0] + 1).tolist()
    out = np.isin(output, idx).reshape(output.shape)
    cleaned = np.where(out, 0, mask)

    return cleaned


def filter_points(x, y, min_distance):
    """
    Removes all but one point if multiple are close-by
    :param x: x-coordinates of points
    :param y: y-coordinates of points
    :param min_distance: minimum distance between points required for them to be both maintained
    :return: filtered points
    """
    points = np.array([[a, b] for a, b in zip(x, y)], dtype=np.int32)

    filtered_points = []
    remaining_points = points.copy()

    while len(remaining_points) > 0:
        current_point = remaining_points[0]
        remaining_points = np.delete(remaining_points, 0, axis=0)
        filtered_points.append(current_point)
        distances = np.linalg.norm(remaining_points - current_point, axis=1)
        remaining_points = remaining_points[distances >= min_distance]

    return np.array(filtered_points)


def remove_points_from_mask(mask, classes):
    """
    Removes predicted pycnidia and rust pustules from the mask. Replaces the relevant pixel values with the average
    of the surrounding pixels. Points need to be transformed separately and added again to the transformed mask.
    :param mask: the mask to remove the points from
    :param classes: ta list with indices of the classes that are represented as points
    :return: mask with key-points removed
    """

    mask = copy.copy(mask)
    for cl in classes:
        idx = np.where(mask == cl)
        y_points, x_points = idx
        for i in range(len(y_points)):
            row, col = y_points[i], x_points[i]
            surrounding_pixels = mask[max(0, row - 1):min(row + 2, mask.shape[0]),
                                 max(0, col - 1):min(col + 2, mask.shape[1])]
            average_value = np.mean(surrounding_pixels)
            mask[row, col] = average_value
    return mask

def select_roi_2(rect, mask):
    x, y, w, h = rect
    roi = np.zeros_like(mask)
    roi[y:y + h, x:x + w] = mask[y:y + h, x:x + w]
    c, _ = cv2.findContours(roi, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    largest_contour = max(c, key=cv2.contourArea)
    roi = np.zeros_like(roi)
    cv2.drawContours(roi, [largest_contour], 0, 255, thickness=cv2.FILLED)
    return roi

def get_bounding_boxes(rect):
    """
    Get bounding boxes of each maintained lesion in a full leaf image
    :param rect: the original rectangle
    :return: Coordinates of the bounding boxes as returned by cv2.boundingRect()
    """
    x, y, w, h = rect
    w = w + 30
    h = h + 30
    x = x - 15
    y = y - 15
    # boxes must not extend beyond the edges of the image
    if x < 0:
        w = w-np.abs(x)
        x = 0
    if y < 0:
        h = h-np.abs(y)
        y = 0
    coords = x, y, w, h

    return coords

def get_pycnidia_maps(mask, resize_factor, bandwidth, kernel):

    # binarize pycnidia, multiply with lesion mask
    pycnidia_binary = np.uint8(np.where(mask == 5, 1, 0))

    # get pycnidia coordinates
    coordinates = np.where(pycnidia_binary == 1)
    coordinates = list(zip(coordinates[0], coordinates[1]))

    if not len(coordinates) > 0:

        color_image_distance = np.zeros_like(mask)
        color_image_density = np.zeros_like(mask)

    else:

        dmap = ndimage.distance_transform_edt(1 - pycnidia_binary)
        # dmap = np.where(dmap > 255, 255, dmap)

        # Normalize the distance map to the range [0, 1]
        norm = Normalize(vmin=dmap.min(), vmax=dmap.max())
        normalized_dmap = norm(dmap)

        # Map the normalized distance map to a colormap
        colormap = plt.cm.viridis  # Change to another colormap if preferred
        color_image_distance = colormap(normalized_dmap)

        # get kernel density esimate
        kde = KernelDensity(bandwidth=bandwidth, kernel=kernel)
        kde.fit(coordinates)

        # get lesion mask
        lesion_mask = remove_points_from_mask(mask=mask, classes=(5, 6))
        lesion_mask = np.where(lesion_mask == 2, 1, 0)
        # lesion_mask = np.uint8(lesion_mask * 255)

        # resize for faster processing
        height = int(lesion_mask.shape[0] / resize_factor)
        width = int(lesion_mask.shape[1] / resize_factor)
        x = np.linspace(0, lesion_mask.shape[1] - 1, width)  # Match resized grid
        y = np.linspace(0, lesion_mask.shape[0] - 1, height)
        x, y = np.meshgrid(x, y)
        grid_coords = np.vstack([y.ravel(), x.ravel()]).T  # Note: (y, x) for consistency

        # Evaluate KDE on the grid
        log_density = kde.score_samples(grid_coords)
        density = np.exp(log_density).reshape(height, width)
        density *= len(coordinates)  # Scale density by the total number of points
        density_rsz = cv2.resize(density, (mask.shape[1], mask.shape[0]), interpolation=cv2.INTER_NEAREST)
        norm = Normalize(vmin=0, vmax=0.004) # max density value was 0.398 across entire data set, but different dist
        normalized_density = norm(density_rsz)

        # Map the normalized density to a colormap
        colormap = plt.cm.plasma
        color_image = colormap(normalized_density)

        # Remove the alpha channel and scale to 0-255 for saving
        color_image_density = (color_image[:, :, :3] * 255).astype(np.uint8)

    return color_image_distance, color_image_density