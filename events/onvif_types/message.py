"""Type module that defines types for ONVIF Messages."""
from __future__ import annotations

from typing import TypedDict


class _SimpleItem(TypedDict):
    Name: str
    Value: str


class _Data(TypedDict):
    SimpleItem: list[_SimpleItem]


class _Value1(TypedDict):
    Data: _Data


class _Message(TypedDict):
    _value_1: _Value1


class _Topic(TypedDict):
    _value_1: str


class Message(TypedDict):
    """ONVIF message that combines the message info with the topic info."""

    Topic: _Topic
    Message: _Message


class Messages(TypedDict):
    """Multiple ONVIF messages inside a list."""

    NotificationMessage: list[Message]
