# Перетворення моделі у OSC.

from pythonosc import udp_client

class OscSender:

    def __init__(
        self,
        host="127.0.0.1",
        port=9000
    ):

        self.client = udp_client.SimpleUDPClient(
            host,
            port
        )

    def send_marker(self, marker):

        c = marker.corners
        r = marker.rvec[0]

        msg = [

            marker.marker_id,

            marker.tx,
            marker.ty,
            marker.tz,

            marker.roll,
            marker.pitch,
            marker.yaw,

            float(r[0]),
            float(r[1]),
            float(r[2]),

            int(c[0][0]),
            int(c[0][1]),

            int(c[1][0]),
            int(c[1][1]),

            int(c[2][0]),
            int(c[2][1]),

            int(c[3][0]),
            int(c[3][1]),
        ]

        self.client.send_message(
            "/marker",
            msg
        )
