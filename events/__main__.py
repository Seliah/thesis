"""Main program for the event implementations."""
# Disable __futures__ import hint as it makes typer unfunctional on python 3.8
# ruff: noqa: FA100

from analysis.app_logging import logger  # noqa: I001

from asyncio import gather
from pathlib import Path
from typing import Annotated, Any, Dict, cast

import onvif
import typer
from httpx import AsyncClient
from lxml import etree
from rich.console import Console

from analysis.camera_info import get_cameras
from analysis.types_adeck.camera import Camera
from analysis.util.tasks import create_task, typer_async
from events.pull_point import pull_point_messages
from events.reactions import handle_message, print_message
from user_secrets import PASS, USER

app = typer.Typer()
console = Console()
client = AsyncClient(timeout=5)

# Use wsdl files of the module to ensure compatibility
# See https://github.com/home-assistant/core/blob/2023.11.2/homeassistant/components/onvif/device.py#L681
WSDL_PATH = f"{Path(onvif.__file__).parent.absolute()}/wsdl/"


async def _get_camera(host: str, port: int):
    logger.info(f"Loading WSDL files from: {WSDL_PATH}")
    camera = onvif.ONVIFCamera(
        host,
        port,
        USER,
        PASS,
        wsdl_dir=WSDL_PATH,
        no_cache=True,
    )
    await camera.update_xaddrs()
    return camera


@app.command()
@typer_async
async def info(
    host: Annotated[str, typer.Argument(help="IP or hostname of the target system.")],
    port: Annotated[int, typer.Argument(help="Port for ONVIF communication.")] = 80,
):
    """Print basic information about the target camera."""
    camera = await _get_camera(host, port)
    mgmt_service = await camera.create_devicemgmt_service()
    console.rule("Device Information")
    console.print(await mgmt_service.GetDeviceInformation())  # pyright: ignore[reportUnknownMemberType]
    console.rule("xaddrs")
    console.print(camera.xaddrs)  # pyright: ignore[reportUnknownMemberType]


@app.command()
@typer_async
async def available(
    host: Annotated[str, typer.Argument(help="IP or hostname of the target system.")],
    port: Annotated[int, typer.Argument(help="Port for ONVIF communication.")] = 80,
):
    """Print available camera event topic groups for target camera."""
    camera = await _get_camera(host, port)
    event_service = await camera.create_events_service()
    console.rule("Event Information")
    event_props = cast(Dict[str, Any], await event_service.GetEventProperties())  # pyright: ignore[reportUnknownMemberType]
    props = cast(list[etree._Element], event_props["TopicSet"]["_value_1"])  # pyright: ignore[reportPrivateUsage]  # noqa: SLF001 we need this type here
    for prop in props:
        console.print(prop.tag)
        # console.print(etree.tostring(prop, pretty_print=True).decode())  # noqa: ERA001
        # console.print([cast(etree._Element, child).tag for child in prop.getchildren()])  # noqa: ERA001


@app.command()
@typer_async
async def pps(
    host: Annotated[str, typer.Argument(help="IP or hostname of the target system.")],
    port: Annotated[int, typer.Argument(help="Port for ONVIF communication.")] = 80,
):
    """Listen for ONVIF events and print them to console."""
    camera = await _get_camera(host, port)
    async for message in pull_point_messages(camera):
        print_message(message)


@app.command()
@typer_async
async def listen(
    host: Annotated[str, typer.Argument(help="IP or hostname of the target system.")],
    port: Annotated[int, typer.Argument(help="Port for ONVIF communication.")] = 80,
):
    """Listen for ONVIF events and run defined reactions."""
    camera = await _get_camera(host, port)
    logger.info("Listening...")
    async for message in pull_point_messages(camera):
        handle_message(message)


@app.command()
@typer_async
async def listen_all(
    port: Annotated[int, typer.Argument(help="Port for ONVIF communication.")] = 80,
):
    """Listen for ONVIF events for every camera and run defined reactions."""
    cameras = [(camera, await _get_camera(camera.address, port)) for camera in await get_cameras()]
    tasks = [_get_task(camera_info, onvif_camera) for (camera_info, onvif_camera) in cameras]
    await gather(*tasks)


def _get_task(camera_info: Camera, onvif_camera: onvif.ONVIFCamera):
    return create_task(
        _handle_messages(camera_info, onvif_camera),
        f'Event handling for "{camera_info}"',
        logger,
        print_exceptions=True,
    )


async def _handle_messages(camera_info: Camera, camera: onvif.ONVIFCamera):
    async for message in pull_point_messages(camera):
        handle_message(message, camera_info)


if __name__ == "__main__":
    app()
