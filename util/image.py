from typing import Any

import cv2
from cv2.typing import MatLike
from numpy import linspace
from numpy.typing import NDArray

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

    # draw vertical lines
    for x in linspace(start=dx, stop=w - dx, num=cols - 1):
        x = int(round(x))
        cv2.line(img, (x, 0), (x, h), color=GRID_COLOR, thickness=GRID_THICKNESS)

    # draw horizontal lines
    for y in linspace(start=dy, stop=h - dy, num=rows - 1):
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
