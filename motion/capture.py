from __future__ import annotations

from asyncio import gather, get_event_loop
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from logging import getLogger
from multiprocessing import Pipe, Process
from multiprocessing.connection import Connection

import cv2
from cv2.typing import MatLike
from numpy import save

import state
from motion.motion import analyze_diff
from motion.read import load_motions
from util.tasks import create_task

loop = get_event_loop()
thread_executor = ThreadPoolExecutor()
process_executor = ProcessPoolExecutor()

_logger = getLogger(__name__)


def capture_motion_sync(source: str, camera_id: str, show: bool):
    capture = cv2.VideoCapture(source)
    reference_frame = None
    while capture.isOpened() and not state.terminating:
        success, frame = capture.read()
        if not success:
            break
        # Get UMat from Matlike to use GPU in following calulcations via OpenCL
        frame_umat = cv2.UMat(frame)
        reference_frame = analyze_diff(frame_umat, reference_frame, camera_id, show)


def capture_subprocess(source: str, conn: Connection, camera_id: str, show: bool):
    capture = cv2.VideoCapture(source)
    _logger.debug(f'Opened video stream for source: "{source}"')
    reference_frame = None
    while capture.isOpened():
        success, frame = capture.read()
        if not success:
            break
        frame_umat = cv2.UMat(frame)
        reference_frame, change_matrix = analyze_diff(frame_umat, reference_frame, camera_id, show)
        if change_matrix is not None:
            conn.send(change_matrix)


def get_change_matrices(source: str, camera_id: str, show: bool):
    output_connection, input_connection = Pipe()
    process = Process(target=capture_subprocess, args=(source, input_connection, camera_id, show))
    process.start()
    while not state.terminating:
        if output_connection.poll():
            yield output_connection.recv()
    cv2.destroyAllWindows()
    process.terminate()
    process.join()


def analyze_motion(source: str, camera_id: str, show: bool):
    # for matrix in get_frames(source, camera_id, show):
    #     # print(matrix.nonzero()[1])
    #     pass
    for matrix in get_change_matrices(source, camera_id, show):
        pass


# def capture_motion_rxpy(source: str, camera_id: str, show: bool):
# capture = cv2.VideoCapture(source)
# if show:
#     from_capture(capture).subscribe(display)
# else:
#     from_capture(capture).subscribe()


def display(frame: MatLike):
    cv2.imshow("Motion Detection", frame)
    cv2.waitKey(1)


async def analyze(source: str, camera_id: str, show: bool):
    # await loop.run_in_executor(executor, capture_motion_sync, source, camera_id, show)
    await loop.run_in_executor(thread_executor, analyze_motion, source, camera_id, show)


async def analyze_sources(sources: dict[str, str], display: str | None = None):
    """Run analysis for all given sources. Save results after termination.

    :param sources: Dictionary that lists the sources as values.
    The keys should be unique identifiers for thes URLs as the analysis results will be saved
    with these keys as IDs.
    :param display: ID for a specific source. When given, the corresponding analysis will be visualized.
    """
    state.motions = load_motions()
    tasks = [
        create_task(
            analyze(source, source_id, source_id == display),
            source_id,
            _logger,
        )
        for source_id, source in sources.items()
    ]
    _logger.info("Analysis is running.")
    await gather(*tasks)
    _logger.info("All analysis tasks terminated.")
    with state.PATH_MOTIONS.open("wb") as f:
        save(f, state.motions)
        _logger.info("Wrote analysis results to disk.")
