#!/usr/bin/env python3

import rospy
from qlin_ms_camera_processing.msg import TrackerStatus

module_name = "qlin_tracker"


class QlinTrackerProvider:
    def __init__(self, module_prefix):
        self.name = rospy.get_name()

        self.pub_tracking_state = rospy.Publisher(module_prefix + "tracking_state", TrackerStatus,
                                                  queue_size=1)

    def set_tracking_state(self, has_lock, pos, f_length, debug=False):
        if not has_lock and debug:
            rospy.logwarn("[" + rospy.get_name() + "]::[qlin_tracker]::Tracker lost lock")

        msg = TrackerStatus(has_lock=has_lock,
                            pos=pos,
                            f_length=f_length)
        self.pub_tracking_state.publish(msg)

    def reset_has_lock(self):
        msg = TrackerStatus(has_lock=False,
                            pos=[0, 0],
                            f_length=[0, 0])
        self.pub_tracking_state.publish(msg)
