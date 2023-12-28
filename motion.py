import cv2
import numpy
from cv2.typing import MatLike
from numpy import concatenate

from user_secrets import URL
from util.image import draw_grid, draw_overlay

cap = cv2.VideoCapture(URL)


def generate_boolean_matrix(image: MatLike, grid_size: tuple[int, int] = (9, 16)):
    # Get the dimensions of the image
    height, width = image.shape

    # Calculate the size of each cell in the grid
    cell_height = height // grid_size[0]
    cell_width = width // grid_size[1]

    # Initialize the boolean matrix
    boolean_matrix = numpy.zeros(grid_size, dtype=bool)

    # Iterate over the cells in the grid
    for i in range(grid_size[0]):
        for j in range(grid_size[1]):
            # Extract the current cell from the image
            cell = image[
                i * cell_height : (i + 1) * cell_height,
                j * cell_width : (j + 1) * cell_width,
            ]

            # Check if the cell contains any non-zero values
            has_values = numpy.any(cell != 0)

            # Update the boolean matrix
            boolean_matrix[i, j] = has_values
    return boolean_matrix


reference_frame = None

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Convert the frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Apply GaussianBlur to reduce noise and improve motion detection
    gray_blurred = cv2.GaussianBlur(gray, (21, 21), 0)

    # # Store the first frame as the reference frame
    if reference_frame is None:
        reference_frame = gray_blurred
        continue

    # Compute the absolute difference between the current frame and the reference frame
    frame_diff = cv2.absdiff(reference_frame, gray_blurred)

    # Apply a threshold to identify regions with significant differences
    _, threshold_diff = cv2.threshold(frame_diff, 30, 255, cv2.THRESH_BINARY)

    # Store the current frame as the reference frame
    reference_frame = gray_blurred
    # Iterate over the contours and perform further processing as needed

    grid = draw_grid(frame.copy(), (9, 16))
    change_matrix = generate_boolean_matrix(threshold_diff, (9, 16))
    overlayed = draw_overlay(grid, change_matrix)

    # Display the results or perform other actions based on motion detection
    top_row = concatenate(
        (frame, cv2.cvtColor(gray_blurred, cv2.COLOR_GRAY2BGR)), axis=1
    )
    bottom_row = concatenate(
        (
            cv2.cvtColor(frame_diff, cv2.COLOR_GRAY2BGR),
            overlayed,
        ),
        axis=1,
    )
    merged = concatenate((top_row, bottom_row), axis=0)
    cv2.imshow("Motion Detection", cv2.resize(merged, (1920, 1080)))
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
