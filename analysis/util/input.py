"""Module for defining asyncio input reading.

See https://stackoverflow.com/questions/35223896/listen-to-keypress-with-asyncio
"""
from __future__ import annotations

import sys
from asyncio import AbstractEventLoop, Queue, ensure_future, get_event_loop


class _Prompt:
    def __init__(self, loop: AbstractEventLoop | None = None):
        self.loop = loop or get_event_loop()
        self.queue: Queue[str] = Queue()
        self.loop.add_reader(sys.stdin, self.got_input)

    def got_input(self):
        ensure_future(self.queue.put(sys.stdin.read(1)), loop=self.loop)

    async def __call__(self):
        return (await self.queue.get()).rstrip("\n")


prompt = _Prompt()
