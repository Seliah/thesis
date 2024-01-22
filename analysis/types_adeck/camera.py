"""Module that defines the type for cameras."""
from typing import Any, Mapping

from pydantic.dataclasses import dataclass

from analysis.types_adeck import BaseType


@dataclass
class Credentials:
    username: str
    password: str


@dataclass
class _Stream:
    token: str
    url: str


@dataclass
class _Streams:
    hd: _Stream
    sd: _Stream


@dataclass
class Camera(BaseType):
    _type = "camera"
    name: str
    uuid: str
    enabled: bool
    protected: bool
    position: int
    address: str
    credentials: Credentials
    scopes: Mapping[str, str]
    streams: _Streams
    has_motiondetection: bool
    use_motion_events: bool
    number: int

    def __repr__(self) -> str:
        return f'{self.name} ("{self.uuid}")'

    @staticmethod
    def repr_raw(json: Any) -> str:
        return f'{json.get("name", "Unknown")} ("{json.get("uuid", "Unknown")}")'

    def __eq__(self, __value: "Camera"):
        return self.uuid == __value.uuid
