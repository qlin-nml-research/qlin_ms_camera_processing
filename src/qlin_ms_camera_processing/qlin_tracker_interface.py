#!/usr/bin/env python3

import rospy
from qlin_ms_camera_processing.msg import TrackerStatus


# from std_msgs.msg import Float64, Bool

class QlinTrackerInterface:
    def __init__(self, module_prefix):
        self.name = rospy.get_name()

        self.has_lock = None
        self.pos_x = None
        self.pos_y = None

        self.sub_tracking_state = rospy.Subscriber(module_prefix + "tracking_state", TrackerStatus,
                                                   self._callback_tracking_state)

    def _callback_tracking_state(self, msg):
        self.has_lock = msg.has_lock
        self.pos_x = msg.pos_x
        self.pos_y = msg.pos_y

    def is_enabled(self):
        if self.has_lock is None:
            return False
        return True

    def has_lock(self):
        return self.has_lock

    def get_pos(self):
        return self.pos_x, self.pos_y
