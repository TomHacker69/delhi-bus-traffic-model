"""
pipeline/live_feed.py
Handles live GTFS-RT vehicle positions feed ingestion.
If GTFS_RT_URL and GTFS_RT_KEY are provided, polls the live protobuf feed,
extracts real-time speeds, and updates the TravelTimeModel in real-time.
Gracefully degrades to historical-only mode when credentials or endpoints are absent.
"""

import os
import time
import asyncio
import logging
from typing import Optional
import httpx
from google.transit import gtfs_realtime_pb2

logger = logging.getLogger("live_feed")
if not logger.handlers:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

GTFS_RT_URL = os.environ.get("GTFS_RT_URL")
GTFS_RT_KEY = os.environ.get("GTFS_RT_KEY")


class GTFSRTLivePoller:
    """
    Polls GTFS-RT feed periodically and updates segment speeds in TravelTimeModel.
    """

    def __init__(self, url: Optional[str] = None, api_key: Optional[str] = None):
        self.url = url or GTFS_RT_URL
        self.api_key = api_key or GTFS_RT_KEY
        self.is_active = bool(self.url and self.api_key)
        self.last_poll_ts: float = 0.0
        self.last_update_count: int = 0
        self.status = "active" if self.is_active else "disabled (historical-only)"

    async def fetch_and_update(self, model) -> int:
        """
        Fetches live GTFS-RT feed, parses protobuf, and updates model live speed ratios.
        Returns number of vehicles/segments updated.
        """
        if not self.is_active:
            return 0

        headers = {
            "User-Agent": "Delhi-Bus-Planner/1.0",
        }
        if self.api_key:
            headers["x-api-key"] = self.api_key
            headers["Authorization"] = f"Bearer {self.api_key}"

        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                resp = await client.get(self.url, headers=headers)
                if resp.status_code != 200:
                    logger.warning(f"GTFS-RT fetch returned HTTP {resp.status_code}")
                    return 0

                feed = gtfs_realtime_pb2.FeedMessage()
                feed.ParseFromString(resp.content)

                updated = 0
                now_ts = time.time()

                for entity in feed.entity:
                    if entity.HasField("vehicle"):
                        v = entity.vehicle
                        pos = v.position
                        speed_ms = pos.speed if pos.HasField("speed") else None
                        route_id = v.trip.route_id if v.HasField("trip") else None

                        if speed_ms is not None and speed_ms >= 0:
                            speed_kmh = float(speed_ms * 3.6)
                            # If route_id is present, update route-associated segments
                            if route_id:
                                # Find active segments for this route
                                for key, rec in model.seg_lookup.items():
                                    seg_id = key[0]
                                    if seg_id.startswith(f"{route_id}_") or seg_id.startswith(f"DTC_{route_id}_"):
                                        model.update_live_speed(seg_id, speed_kmh, now_ts)
                                        updated += 1

                self.last_poll_ts = now_ts
                self.last_update_count = updated
                logger.info(f"Live GTFS-RT updated {updated} segment speeds at {now_ts}")
                return updated

        except Exception as e:
            logger.warning(f"Failed to fetch/parse GTFS-RT feed: {e}")
            return 0

    async def run_loop(self, model, poll_interval_s: int = 30):
        """Runs the continuous polling loop in the background."""
        if not self.is_active:
            logger.info("GTFS-RT URL or KEY missing. Live poller idle; running in historical-only mode.")
            return

        logger.info(f"Starting GTFS-RT live poller background loop (interval: {poll_interval_s}s)")
        while True:
            try:
                await self.fetch_and_update(model)
            except asyncio.CancelledError:
                logger.info("Live poller loop cancelled.")
                break
            except Exception as e:
                logger.error(f"Unexpected error in live poller: {e}")
            await asyncio.sleep(poll_interval_s)
