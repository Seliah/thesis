from asyncio import gather, get_event_loop
from concurrent.futures import Executor, ThreadPoolExecutor
from logging import DEBUG, basicConfig, getLogger
from pathlib import Path
from signal import Signals

import cv2
from httpx import AsyncClient
from numpy import save

import state
from motion.capture import capture_motion, motions
from types_adeck import parse_all
from types_adeck.camera import Camera
from user_secrets import C
from util.input import prompt
from util.tasks import create_task

client = AsyncClient(verify=C, timeout=5)

CAMERA_URL: str = "https://adeck:8442/cameras?enabled=true"

basicConfig(level=DEBUG)
_logger = getLogger()


async def get_cameras():
    result = await client.get(CAMERA_URL)
    return parse_all(Camera, result.json()["result"])


def get_rtsp_url(camera: Camera):
    """Get the RTSP URL for recordings with credentials."""
    rtsp_url = camera.streams.sd.url
    # Add credentials to rtsp_url
    credentials = camera.credentials
    return rtsp_url.replace(
        "rtsp://", f"rtsp://{credentials.username}:{credentials.password}@"
    )


loop = get_event_loop()


async def capture(camera: Camera, executor: Executor):
    url = get_rtsp_url(camera)
    capture = cv2.VideoCapture(url)
    await loop.run_in_executor(executor, capture_motion, capture, camera)


async def _service_terminate(signal: Signals):
    _logger.info(
        f'Received signal "{signal.name}" ({signal}). Service is shutting down.'
    )
    state.terminating = True


async def _main():
    # set_termination_handler(_service_terminate)
    cameras = await get_cameras()
    with ThreadPoolExecutor(None, "Capture") as executor:
        tasks = [
            create_task(capture(camera, executor), camera.name, _logger)
            for camera in cameras
        ]
        _logger.info("Running")
        await prompt()
        _logger.info("Got input, exiting...")
        state.terminating = True
        await gather(*tasks)
        _logger.info("done")
        with Path("motions.npy").open("wb") as f:
            save(f, motions)
            _logger.info("Wrote file!")


if __name__ == "__main__":
    # set_termination_handler(_service_terminate)
    # run(_main())
    loop.run_until_complete(_main())

    # _logger.info("Service is running.")
    # loop.run_forever()
    # _logger.info("Service shut down successfully.")
