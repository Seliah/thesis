from __future__ import annotations

import signal
import threading
from asyncio import gather, get_event_loop
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from logging import getLogger
from multiprocessing import Manager, Pipe
from typing import TYPE_CHECKING, Any

from cv2 import VideoCapture
from cv2.typing import MatLike
from numpy import asanyarray, save
from reactivex import merge
from reactivex import operators as ops
from reactivex.subject import Subject

from analysis import definitions, state
from analysis.util.rx import from_capture
from analysis.util.tasks import create_task
from analysis.vision import analyses
from analysis.vision.analyses import Analysis

if TYPE_CHECKING:
    from multiprocessing.connection import Connection

loop = get_event_loop()

_logger = getLogger(__name__)

subjects: set[Subject[Any]] = set()
thread_executor = ThreadPoolExecutor(thread_name_prefix="Parser")
process_executor = ProcessPoolExecutor(16)


def get_merged_output(show: bool, frames: Subject[MatLike], analyses: dict[str, Analysis[Any]]):
    output_streams = [get_output_stream(name, analysis, frames, show) for name, analysis in analyses.items()]
    return merge(*output_streams)


def get_output_stream(name: str, analysis: Analysis[Any], frames: Subject[MatLike], show: bool):
    return analysis.analyze(frames, show).pipe(
        ops.map(lambda result: (name, result)),
    )


def capture_sync(
    source: str,
    show: bool,
    termination_event: threading.Event,
    analyses: dict[str, Analysis[Any]],
    conn: Connection,
):
    if not definitions.IS_SERVICE:
        signal.signal(signal.SIGINT, signal.SIG_IGN)

    frames = Subject[MatLike]()
    subjects.add(frames)

    get_merged_output(show, frames, analyses).subscribe(conn.send)

    capture_stream = from_capture(VideoCapture(source), termination_event)
    capture_stream.subscribe(frames)


async def capture(source: str, show: bool, input_connection: Connection, event: threading.Event):
    await loop.run_in_executor(
        process_executor,
        capture_sync,
        source,
        show,
        event,
        analyses.analyses,
        input_connection,
    )


def parse_results_sync(output_connection: Connection, camera_id: str):
    while not state.terminating.is_set():
        if output_connection.poll(1):
            output: tuple[str, Any] = output_connection.recv()
            [name, result] = output
            analyses.analyses[name].parse(result, camera_id)


async def parse_results(output_connection: Connection, camera_id: str):
    await loop.run_in_executor(
        thread_executor,
        parse_results_sync,
        output_connection,
        camera_id,
    )


async def analyze(source: str, camera_id: str, show: bool, event: threading.Event):
    output_connection, input_connection = Pipe()
    coro = capture(source, show, input_connection, event)
    create_task(coro, "Subprocess handler task", _logger)
    await parse_results(output_connection, camera_id)


async def analyze_sources(sources: dict[str, str], display: str | None = None):
    """Run analysis for all given sources. Save results after termination.

    :param sources: Dictionary that lists the sources as values.
    The keys should be unique identifiers for thes URLs as the analysis results will be saved
    with these keys as IDs.
    :param display: ID for a specific source. When given, the corresponding analysis will be visualized.
    """
    with Manager() as manager:
        event = manager.Event()
        tasks = [
            create_task(
                analyze(source, source_id, source_id == display, event),
                source_id,
                _logger,
            )
            for source_id, source in sources.items()
        ]
        _logger.info("Analysis is running.")
        await state.terminating.wait()
        event.set()
        # Cancel future tasks
        process_executor.shutdown(wait=False, cancel_futures=True)
        await gather(*tasks)
    _logger.info("All analysis tasks terminated.")
    with definitions.PATH_MOTIONS.open("wb") as f:
        # TODO check if this works
        save(f, asanyarray(state.motions))
        _logger.info("Wrote analysis results to disk.")
