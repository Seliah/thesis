"""Module for implementing the logic to get camera information.

This is done via HTTP communication with the adeck VMS.
"""
from httpx import AsyncClient

from analysis.definitions import PATH_SETTINGS
from analysis.types_adeck import parse_all, settings
from analysis.types_adeck.camera import Camera
from user_secrets import CAMERA_URL, C

client = AsyncClient(verify=C, timeout=5)
EXCLUDES = settings.load(PATH_SETTINGS).excludes


async def get_cameras():
    """Get all cameras from the adeck VMS."""
    result = await client.get(CAMERA_URL)
    cameras = parse_all(Camera, result.json()["result"])
    return [*filter(_is_not_excluded, cameras)]


def _is_not_excluded(camera: Camera):
    return camera.uuid not in EXCLUDES


def get_rtsp_url(camera: Camera):
    """Get the RTSP URL for recordings with credentials."""
    rtsp_url = camera.streams.sd.url
    # Add credentials to rtsp_url
    credentials = camera.credentials
    return rtsp_url.replace(
        "rtsp://",
        f"rtsp://{credentials.username}:{credentials.password}@",
    )


async def get_sources():
    """Get all cameras from the adeck VMS, mapped by their ID."""
    return {camera.uuid: get_rtsp_url(camera) for camera in await get_cameras()}
