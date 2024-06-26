"""Module for providing standardized asyncio task usage tools."""
from __future__ import annotations

import asyncio
from asyncio import Task, get_event_loop
from functools import wraps
from typing import TYPE_CHECKING, Any, Callable, Coroutine, TypeVar

if TYPE_CHECKING:
    from logging import Logger

T = TypeVar("T")
LOG_TASKS = True

loop = get_event_loop()


def typer_async(f: Callable[..., Any]):
    """Run a function in the current event loop.

    This can be used to use an async function as a typer command.
    This code was orignally formulated in typer's issue

    See https://github.com/tiangolo/typer/issues/85#issuecomment-1365871959
    """

    @wraps(f)
    def wrapper(*args: ..., **kwargs: ...):
        return loop.run_until_complete(f(*args, **kwargs))

    return wrapper


def _task_done(
    task: Task[T],
    logger: Logger,
    on_done: Callable[[Task[T]], Any] | None,
    print_exceptions: bool,
):
    try:
        task.result()
        if __debug__ and LOG_TASKS:
            logger.debug(f'Task "{task.get_name()}" finished.')
    except asyncio.CancelledError:
        logger.debug(f'Task "{task.get_name()}" got cancelled:')
    except Exception:  # noqa: BLE001 this is intend to catch blind as we don't have insight into overlying calls
        if print_exceptions:
            logger.exception(f'Exception raised by task "{task.get_name()}."')
    if on_done is not None:
        on_done(task)


def create_task(
    coroutine: Coroutine[Any, Any, T],
    name: str,
    logger: Logger,
    on_done: Callable[[Task[T]], Any] | None = None,
    print_exceptions: bool = False,
) -> Task[T]:
    """Create a task with exception printing.

    Global usage of this function will ensure that exceptions in tasks are always correctly displayed.
    See https://quantlane.com/blog/ensure-asyncio-task-exceptions-get-logged/
    """
    if __debug__ and LOG_TASKS:
        logger.debug(f'Creating Task "{name}".')
    loop = get_event_loop()
    task = loop.create_task(coroutine, name=name)
    task.add_done_callback(
        lambda task: _task_done(task, logger, on_done, print_exceptions),
    )
    return task
