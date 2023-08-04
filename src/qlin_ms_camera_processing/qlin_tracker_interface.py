#!/usr/bin/env python3
import numpy as np
import rospy
from qlin_ms_camera_processing.msg import TrackerStatus


# from std_msgs.msg import Float64, Bool

class QlinTrackerInterface:
    def __init__(self, module_prefix):
        self.name = rospy.get_name()

        self.has_lock = None
        self.pos = None
        self.f_length = None

        self.sub_tracking_state = rospy.Subscriber(module_prefix + "tracking_state", TrackerStatus,
                                                   self._callback_tracking_state)

    def _callback_tracking_state(self, msg):
        self.has_lock = msg.has_lock
        self.pos = msg.pos
        self.f_length = msg.f_length

    def is_enabled(self):
        if self.has_lock is None:
            return False
        return True

    def get_has_lock(self):
        if self.has_lock:
            self.has_lock = False
            return True
        return self.has_lock

    def get_pos(self):
        return np.array(self.pos)

    def get_f_length(self):
        return self.f_length


    # to get the x, y, position, should be simple pos * object_d
    # object_d is basically, camera_seperation_distance- f_distance
