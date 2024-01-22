"""Module for implementing reusable functionality for image data.

This can for example be functions for visualizations or other image edits/analysis functions.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Any, Tuple

import cv2
from numpy import array, linspace
from scipy.spatial.distance import euclidean

if TYPE_CHECKING:
    from cv2.typing import MatLike
    from numpy.typing import NDArray

Point = Tuple[float, float]
RectPoints = Tuple[Point, Point, Point, Point]

GRID_COLOR = (0, 255, 0)
GRID_THICKNESS = 1
ALPHA = 0.2


def draw_grid(img: MatLike, grid_shape: tuple[int, int]):
    """Draw a grid onto a given image.

    Source: https://gist.github.com/mathandy/389ddbad48810d188bdc997c3a1dab0c
    """
    h, w, _ = img.shape
    rows, cols = grid_shape
    dy, dx = h / rows, w / cols

    space_x = linspace(start=dx, stop=w - dx, num=cols - 1)
    space_y = linspace(start=dy, stop=h - dy, num=rows - 1)

    # draw vertical lines
    for x in space_x:
        x = int(round(x))
        cv2.line(img, (x, 0), (x, h), color=GRID_COLOR, thickness=GRID_THICKNESS)

    # draw horizontal lines
    for y in space_y:
        y = int(round(y))
        cv2.line(img, (0, y), (w, y), color=GRID_COLOR, thickness=GRID_THICKNESS)

    return img


def draw_overlay(img: MatLike, change_matrix: NDArray[Any]):
    height_image, width_image, _ = img.shape
    height_mat, width_mat = change_matrix.shape
    cell_height = height_image // height_mat
    cell_width = width_image // width_mat
    # See https://stackoverflow.com/questions/60456616/how-do-i-make-an-inverse-filled-transparent-rectangle-with-opencv
    output = img.copy()

    for y in range(height_mat):
        for x in range(width_mat):
            if change_matrix[y][x]:
                left = int(x * cell_width)
                right = int(x * cell_width) + cell_width
                top = int(y * cell_height)
                bottom = int(y * cell_height) + cell_height
                cv2.rectangle(
                    img,
                    (left, top),
                    (right, bottom),
                    GRID_COLOR,
                    -1,
                )
                cv2.addWeighted(img, ALPHA, output, 1 - ALPHA, 0, output)
    return output


def warp(image: MatLike, points: RectPoints):
    """Get warped subimage of given image, bound by a given 4-corner polygon.

    See https://pyimagesearch.com/2014/08/25/4-point-opencv-getperspective-transform-example/
    See https://theailearner.com/tag/cv2-warpperspective/
    """
    width_max, height_max = _get_max_size(points)
    src = array(points, dtype="float32")
    dst = array(
        [[0.0, 0.0], [width_max - 1, 0.0], [width_max - 1, height_max - 1], [0, height_max - 1]],
        dtype="float32",
    )
    transform = cv2.getPerspectiveTransform(src, dst)
    return cv2.warpPerspective(image, transform, (width_max, height_max), flags=cv2.INTER_LINEAR)


def _get_max_size(points: RectPoints):
    (top_left, top_right, bottom_right, bottom_left) = points
    # See https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.distance.euclidean.html#scipy.spatial.distance.euclidean
    width_top = euclidean(top_left, top_right)
    width_bottom = euclidean(bottom_left, bottom_right)
    width_max = max(int(width_top), int(width_bottom))
    height_left = euclidean(top_left, bottom_left)
    height_right = euclidean(top_right, bottom_right)
    height_max = max(int(height_left), int(height_right))
    return width_max, height_max
