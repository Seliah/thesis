from __future__ import annotations

from asyncio import gather, get_event_loop
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from logging import getLogger
from multiprocessing import Pipe
from multiprocessing.connection import Connection
from threading import Event

import cv2
from numpy import save
from reactivex.operators import do_action, pairwise, throttle_first
from reactivex.operators import map as map_op

import state
from motion.motion import analyze_diff, display, prepare, update_global_matrix
from motion.read import load_motions
from util.rx import from_capture
from util.tasks import create_task

FPS = 5
TIME_PER_FRAME = 1 / FPS

loop = get_event_loop()
thread_executor = ThreadPoolExecutor()
process_executor = ProcessPoolExecutor(16)

_logger = getLogger(__name__)


def analyze_motion(source: str, show: bool, termination_event: Event):
    capture = cv2.VideoCapture(source)
    return from_capture(capture, termination_event).pipe(
        # Apply FPS
        throttle_first(TIME_PER_FRAME),
        # Apply image preparation for analysis
        map_op(prepare),
        # Keep the previous frame for diff
        pairwise(),
        map_op(lambda pair: analyze_diff(pair[1][0], pair[1][1], pair[0][1])),
        # Display
        do_action(lambda t: display(t[0], t[1], t[2], t[3]) if show else None),
        map_op(lambda t: t[3]),
    )


def capture_motion_sync(source: str, show: bool, termination_event: Event, conn: Connection):
    analyze_motion(source, show, termination_event).subscribe(conn.send)


async def capture_motion(source: str, camera_id: str, show: bool, input_connection: Connection):
    await loop.run_in_executor(
        process_executor,
        capture_motion_sync,
        source,
        show,
        state.termination_event,
        input_connection,
    )


def process_results_sync(output_connection: Connection, camera_id: str):
    while not state.terminating:
        if output_connection.poll(1):
            change_matrix = output_connection.recv()
            update_global_matrix(state.motions, change_matrix, state.GRID_SIZE, camera_id)


async def process_results(output_connection: Connection, camera_id: str):
    await loop.run_in_executor(
        thread_executor,
        process_results_sync,
        output_connection,
        camera_id,
    )


async def analyze(source: str, camera_id: str, show: bool):
    output_connection, input_connection = Pipe()
    coro = capture_motion(source, camera_id, show, input_connection)
    create_task(coro, "Subprocess handler task", _logger)
    await process_results(output_connection, camera_id)


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
