#!/usr/bin/python3
import rospy
import os, sys, time, datetime, traceback
from qlin_ms_camera_processing import QlinTrackerProvider


def qlin_tracker_recevier_main(_name, _config):
    rospy.logwarn("[" + rospy.get_name() + "]::Running")

    tracker_provider = QlinTrackerProvider(_config['ns_prefix'])

    rospy.spin()

    pass


if __name__ == '__main__':

    rospy.init_node("camera_tracking_receiver_node",
                    disable_signals=True,
                    anonymous=False)

    name = rospy.get_name()
    params = rospy.get_param(name)

    config = {}

    try:
        config['port'] = params['port']
        config['ns_prefix'] = params['ns_prefix']

        rospy.loginfo("[" + name + "]:: Parameter load OK.")
    except Exception as e:

        rospy.logerr(name + ": initialization ERROR on parameters")
        rospy.logerr(name + ": " + str(e))
        rospy.signal_shutdown(name + ": initialization ERROR on parameters")
        exit()

    namespace = rospy.get_namespace()

    qlin_tracker_recevier_main(name, config)
