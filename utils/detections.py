"""Useful functions for image processing and detection using OpenCV"""


import cv2
import numpy as np


def find_corners(img):
    """Finds harris corners"""
    corners = cv2.cornerHarris(img, 5, 3, 0.1)
    corners = cv2.dilate(corners, None)
    corners = cv2.threshold(corners, 0.01 * corners.max(), 255, 0)[1]
    corners = corners.astype(np.uint8)
    _, labels, stats, centroids = cv2.connectedComponentsWithStats(
        corners, connectivity=4)
    # For some reason, stats yielded better results for
    # corner detection than centroids. This might have
    # something to do with sub-pixel accuracy. 
    # Check issue #10130 on opencv
    return stats


def contoured_bbox(img):
    """Returns bbox of contoured image"""
    contours, hierarchy = cv2.findContours(img, 1, 2)
    # Largest object is whole image,
    # second largest object is the ROI
    sorted_cntr = sorted(contours, key=lambda cntr: cv2.contourArea(cntr))
    return cv2.boundingRect(sorted_cntr[-2])


def preprocess_input(img):
    """Preprocess image to match model's input shape for shape detection"""
    img = cv2.resize(img, (32, 32))
    # Expand for channel_last and batch size, respectively
    img = np.expand_dims(img, axis=-1)
    img = np.expand_dims(img, axis=0)
    return img.astype(np.float32) / 255
