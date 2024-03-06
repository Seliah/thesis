"""Module to define handlers for specific messages.

A handler is a callback function that is specific to a message topic.
Handlers a kept in a dictionary, mapped by the corresponding topic.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

from rich.console import Console

from analysis.app_logging import logger

if TYPE_CHECKING:
    from analysis.types_adeck.camera import Camera
    from events.onvif_types.message import Message

console = Console()


def _handle_queue(event: Message, camera: Camera | None = None):
    queue_detected = _truthy(_get_value(event))
    log_message = "Queue detected!" if queue_detected else "No queue detected."
    logger.info(f"{_get_label(camera)} - {log_message}")


def _handle_crossing(_event: Message, camera: Camera | None = None):
    logger.info(f"{_get_label(camera)} - Line crossed!")


def _handle_tamper(event: Message, camera: Camera | None = None):
    tamper_detected = _truthy(_get_value(event))
    log_message = "Tampering detected!" if tamper_detected else "No tampering detected."
    logger.info(f"{_get_label(camera)} - {log_message}")


def _handle_intrusion(event: Message, camera: Camera | None = None):
    intrusion_detected = _truthy(_get_value(event))
    log_message = "Intrusion detected!" if intrusion_detected else "No intrusion detected."
    logger.info(f"{_get_label(camera)} - {log_message}")


def _truthy(value: str):
    return value in ("true", "1")


def _get_label(camera: Camera | None = None):
    return str(camera) if camera is not None else "Unkown Camera"


ignore = {"tns1:Device/tnsaxis:IO/VirtualInput"}

handlers = {
    # Queue detection topics Axis
    "tnsaxis:CameraApplicationPlatform/ObjectAnalytics/Device1Scenario1Threshold": _handle_queue,
    "tnsaxis:CameraApplicationPlatform/ObjectAnalytics/Device1Scenario2Threshold": _handle_queue,
    "tnsaxis:CameraApplicationPlatform/ObjectAnalytics/Device1Scenario3Threshold": _handle_queue,
    # Line cross Hik
    "tns1:RuleEngine/LineDetector/Crossed": _handle_crossing,
    # Tampering topic Hik
    "tns1:RuleEngine/TamperDetector/Tamper": _handle_tamper,
    # Tampering topic Axis
    "tns1:VideoSource/tnsaxis:Tampering": _handle_tamper,
    # Region monitoring (intrusion detection) Hik
    "tns1:RuleEngine/FieldDetector/ObjectsInside": _handle_intrusion,
}


def handle_message(message: Message, camera: Camera | None = None):
    """Run a handler for the given message, if defined."""
    topic = _get_topic(message)
    if topic not in ignore:
        handler = handlers.get(topic, None)
        if handler is not None:
            handler(message, camera)


def print_message(message: Message):
    """Print the given messages topic and data, if a handler is defined."""
    topic = _get_topic(message)
    if topic not in ignore:
        logger.debug(f'Message on topic "{topic}"')
        if str(topic) in handlers:
            # logger.debug(f"Message: {message}")  # noqa: ERA001
            logger.debug(f'Message data: {message["Message"]["_value_1"]["Data"]}')


def _get_topic(event: Message):
    return event["Topic"]["_value_1"]


def _get_value(event: Message):
    return event["Message"]["_value_1"]["Data"]["SimpleItem"][0]["Value"]
