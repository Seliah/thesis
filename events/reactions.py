"""Module to define handlers for specific messages.

A handler is a callback function that is specific to a message topic.
Handlers a kept in a dictionary, mapped by the corresponding topic.
"""

from rich.console import Console

from analysis.app_logging import logger
from events.onvif_types.message import Message

console = Console()


def _handle_queue(event: Message):
    value = event["Message"]["_value_1"]["Data"]["SimpleItem"][0]["Value"]
    queue_detected = bool(int(value))
    log_message = "Queue detected!" if queue_detected else "No queue detected."
    logger.info(log_message)


def _handle_crossing(_event: Message):
    logger.info("Line crossed!")


def _handle_tamper(event: Message):
    value = event["Message"]["_value_1"]["Data"]["SimpleItem"][0]["Value"]
    tamper_detected = value == "true"
    log_message = "Tampering detected!" if tamper_detected else "No tampering detected."
    logger.info(log_message)


ignore = {"tns1:Device/tnsaxis:IO/VirtualInput"}

handlers = {
    "tnsaxis:CameraApplicationPlatform/ObjectAnalytics/Device1Scenario1Threshold": _handle_queue,
    "tns1:RuleEngine/LineDetector/Crossed": _handle_crossing,
    "tns1:RuleEngine/TamperDetector/Tamper": _handle_tamper,
}


def handle_message(message: Message):
    """Run a handler for the given message, if defined."""
    topic = _get_topic(message)
    if topic not in ignore:
        handler = handlers.get(topic, None)
        if handler is not None:
            handler(message)


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
