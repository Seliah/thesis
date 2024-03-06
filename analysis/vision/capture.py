"""Module for top level analysis functionality.

This module implements logic for analyzing multiple sources in parallel on multiple processes and threads.
"""
from __future__ import annotations

import signal
from asyncio import gather, get_event_loop, wait_for
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from multiprocessing import Manager, Pipe
from typing import TYPE_CHECKING, Any

from cv2 import (
    VideoCapture,
)
from cv2.typing import MatLike
from reactivex import merge
from reactivex import operators as ops
from reactivex.subject import Subject

from analysis import definitions, state
from analysis.app_logging import logger
from analysis.util.rx import from_capture
from analysis.util.tasks import create_task
from analysis.vision.analyses import Analysis, analyses

if TYPE_CHECKING:
    import threading
    from multiprocessing.connection import Connection

loop = get_event_loop()


subjects: set[Subject[Any]] = set()
thread_executor = ThreadPoolExecutor(thread_name_prefix="ParseThread")
process_executor = ProcessPoolExecutor(16)


async def analyze_sources(sources: dict[str, str], display: str | None = None):
    """Run analysis for all given sources until termination event. Save results after termination.

    :param sources: Dictionary that lists the sources as values.
    The keys should be unique identifiers for thes URLs as the analysis results will be saved
    with these keys as IDs.
    :param display: ID for a specific source. When given, the corresponding analysis will be visualized.
    """
    with Manager() as manager:
        event = manager.Event()
        tasks = [
            create_task(
                _analyze_source(source, source_id, source_id == display, event),
                source_id,
                logger,
            )
            for source_id, source in sources.items()
        ]
        logger.info("Analysis is running.")
        await state.terminating.wait()
        logger.debug("Program terminating, shutting down analysis processes.")
        event.set()
        # Cancel future tasks
        process_executor.shutdown(wait=False, cancel_futures=True)
        await gather(*tasks)
        logger.info("All analysis processes terminated.")
        # Run all on_termination callbacks defined by the analyses
        callbacks = [callback for analysis in analyses.values() if (callback := analysis.on_termination) is not None]
        for termination_callback in callbacks:
            termination_callback()


async def _analyze_source(source: str, source_id: str, visualize: bool, event: threading.Event):
    """Analyze the given source.

    This will start a analysis process and a result parsing thread using the module level executors.
    """
    output_connection, input_connection = Pipe()
    # Run capture in multiple processes in background (via asyncio task)
    capture_future = loop.run_in_executor(
        process_executor,
        _capture,
        source,
        visualize,
        event,
        analyses,
        input_connection,
    )
    create_task(
        wait_for(capture_future, timeout=None),
        f'Capture and analysis task for source "{source}"',
        logger,
    )
    # Run parsing in multiple threads in foreground
    await loop.run_in_executor(
        thread_executor,
        _parse,
        output_connection,
        source_id,
    )


def _capture(
    source: str,
    visualize: bool,
    termination_event: threading.Event,
    analyses: dict[str, Analysis[Any]],
    conn: Connection,
):
    """Capture video feed for given source and run all given analyses on it.

    This will set up only one input to minimize the I/O usage for camera and this machine.
    Analysis results are all send over the given connection with an analysis identifier prefix to enable parsing.
    """
    logger.debug(f'Starting video capture and analysis for source "{source}".')
    if not definitions.IS_SERVICE:
        signal.signal(signal.SIGINT, signal.SIG_IGN)

    frames = Subject[MatLike]()
    subjects.add(frames)

    _get_merged_output(frames, analyses, visualize).subscribe(conn.send, on_error=logger.exception)

    capture = VideoCapture()
    capture.open(source)
    logger.debug(f'Video capture initialized for source "{source}". Backend: {capture.getBackendName()}')
    capture_stream = from_capture(capture, termination_event)
    capture_stream.subscribe(frames, logger.exception)


def _get_merged_output(frames: Subject[MatLike], analyses: dict[str, Analysis[Any]], visualize: bool):
    """Create observalbe that combines all analyses.

    This uses the analysis observable that is defined for each analysis and RxPY merge.
    """
    output_streams = [_get_output_stream(frames, analysis, name, visualize) for name, analysis in analyses.items()]
    return merge(*output_streams)


def _get_output_stream(frames: Subject[MatLike], analysis: Analysis[Any], analysis_name: str, visualize: bool):
    """Get the analysis result observable defined by the analysis.

    This will also prefix the results with the analysis ID.
    """
    return analysis.analyze(frames, visualize).pipe(
        ops.map(lambda result: (analysis_name, result)),
    )


def _parse(output_connection: Connection, camera_id: str):
    """Parse the analysis results, send over the given connection.

    This can be used to combine all data in a single heap to make it usable by an API or writing process.
    """
    while not state.terminating.is_set():
        if output_connection.poll(1):
            output: tuple[str, Any] = output_connection.recv()
            [name, result] = output
            analyses[name].parse(result, camera_id)
