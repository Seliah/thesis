from httpx import AsyncClient

from types_adeck import parse_all
from types_adeck.camera import Camera
from user_secrets import CAMERA_URL, C

client = AsyncClient(verify=C, timeout=5)


async def get_cameras():
    result = await client.get(CAMERA_URL)
    return parse_all(Camera, result.json()["result"])


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
    return {camera.uuid: get_rtsp_url(camera) for camera in await get_cameras()}
