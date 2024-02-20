"""Type module that defines types for ONVIF Subscriptions."""
from __future__ import annotations

from typing import TypedDict


class _Address(TypedDict):
    _value_1: str


class ReferenceParameters(TypedDict):
    """Metadata for the reference to an ONVIF pull point subscription."""

    _value_1: str


class _SubscriptionReference(TypedDict):
    Address: _Address
    ReferenceParameters: ReferenceParameters | None


class Subscription(TypedDict):
    """ONVIF pull point subscription with reference."""

    SubscriptionReference: _SubscriptionReference
