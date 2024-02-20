"""Module for defining a zeep plugin that adds a header to messages."""
from __future__ import annotations

from typing import TYPE_CHECKING

from typing_extensions import override
from zeep import Plugin

if TYPE_CHECKING:
    from lxml import etree

ENVELOPE_NAMESPACE = {"soap-env": "http://www.w3.org/2003/05/soap-envelope"}


class HeaderPlugin(Plugin):
    """Zeep plugin to add a header to communication messages."""

    @override  # pyright: ignore[reportUntypedFunctionDecorator]
    def __init__(self, header: str) -> None:
        self.header = header

    @override  # pyright: ignore[reportUntypedFunctionDecorator]
    def egress(self, envelope: etree._Element, http_headers: dict[str, str], *_):  # pyright: ignore[reportPrivateUsage] we need this type here  # noqa: ANN002
        headers = envelope.find("soap-env:Header", namespaces=ENVELOPE_NAMESPACE)  # pyright: ignore[reportUnknownMemberType]
        if headers is not None:
            headers.insert(0, self.header)
        return envelope, http_headers
