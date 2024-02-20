"""Module for implementing ONVIF PullPoint communication logic."""
from datetime import timedelta
from typing import cast

import onvif
from onvif.util import normalize_url

from events.header import HeaderPlugin
from events.onvif_types.message import Messages
from events.onvif_types.subscription import ReferenceParameters, Subscription

SUBSCRIBE_PARAMS = {"InitialTerminationTime": f"PT{60}S"}
PULL_PARAMS = {
    "MessageLimit": 100,
    "Timeout": timedelta(seconds=60),
}


async def pull_point_messages(camera: onvif.ONVIFCamera):
    """Subscribe to ONVIF messages and yield them."""
    subscription = await _subscribe(camera)
    pullpoint_service = await camera.create_pullpoint_service()

    # Support AXIS cameras (they need this auth header)
    if (params := subscription["SubscriptionReference"]["ReferenceParameters"]) is not None:
        _add_header(params, pullpoint_service)

    while True:
        messages = cast(Messages, await pullpoint_service.PullMessages(PULL_PARAMS))  # pyright: ignore[reportUnknownMemberType]
        for message in messages["NotificationMessage"]:
            yield message
    # Maybe another step can be added in the future: the periodic renewal of the subscription


async def _subscribe(camera: onvif.ONVIFCamera):
    event_service = await camera.create_events_service()
    subscription = cast(Subscription, await event_service.CreatePullPointSubscription(SUBSCRIBE_PARAMS))  # pyright: ignore[reportUnknownMemberType]
    camera.xaddrs["http://www.onvif.org/ver10/events/wsdl/PullPointSubscription"] = normalize_url(  # pyright: ignore[reportUnknownMemberType]
        subscription["SubscriptionReference"]["Address"]["_value_1"],
    )
    return subscription


def _add_header(params: ReferenceParameters, pps_service: onvif.ONVIFService):
    """Add auth header to service messages."""
    token = params["_value_1"][0]
    client = pps_service.zeep_client
    if client is not None:
        client.plugins.append(HeaderPlugin(token))  # pyright: ignore[reportUnknownMemberType]
