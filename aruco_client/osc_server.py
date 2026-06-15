import asyncio
import functools
import numpy as np

from pythonosc.dispatcher import Dispatcher
from pythonosc.osc_server import AsyncIOOSCUDPServer

from .models import Marker

class OSCServer:
    def __init__(self, ip, port, markers_dict, scale_x=1.0, scale_y=1.0):
        self.ip = ip
        self.port = port
        self.markers = markers_dict  # This is a reference to the main markers dictionary

        # Create a partial function with the necessary arguments for the handler
        handler_with_context = functools.partial(
            self.marker_handler, 
            markers_ref=self.markers,
            scale_x=scale_x, 
            scale_y=scale_y
        )
        
        dispatcher = Dispatcher()
        dispatcher.map("/marker", handler_with_context)
        
        self.server = AsyncIOOSCUDPServer(
            (self.ip, self.port), dispatcher, asyncio.get_event_loop()
        )
        self.transport = None
        self.protocol = None

    async def start(self):
        """Starts the OSC server."""
        self.transport, self.protocol = await self.server.create_serve_endpoint()
        print(f"OSC Server started on {self.ip}:{self.port}")

    def close(self):
        """Closes the server transport."""
        if self.transport:
            self.transport.close()
        print("OSC Server stopped.")

    @staticmethod
    def marker_handler(address, *args, markers_ref, scale_x, scale_y):
        """
        Static method to handle incoming OSC messages.
        It updates the shared markers dictionary.
        """
        marker_id = args[0]
        tx, ty, tz = args[1:4]
        roll, pitch, yaw = args[4:7]
        raw_corners = np.array(args[7:], dtype=np.float32).reshape((4, 2))
        
        scaled_corners = (raw_corners * [scale_x, scale_y]).astype(np.int32)
        
        if marker_id in markers_ref:
            markers_ref[marker_id].update(tx, ty, tz, roll, pitch, yaw, scaled_corners)
        else:
            markers_ref[marker_id] = Marker(marker_id, tx, ty, tz, roll, pitch, yaw, scaled_corners)
