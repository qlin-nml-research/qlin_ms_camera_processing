#!/usr/bin/python3
import math

import numpy as np
import rospy
import os, sys, time, datetime, traceback
from qlin_ms_camera_processing import QlinTrackerProvider

from PyQt5.QtCore import QByteArray, qChecksum, QDataStream, QIODevice
from PyQt5.QtNetwork import QUdpSocket, QHostAddress

incoming_udp_format = [('pos', 'float', 2), ('has_lock', 'int', 1), ('focal_length', 'float', 2)]

timed_out = 1.0  # sec


def qlin_tracker_recevier_main(_name, _config):
    rospy.logwarn("[" + rospy.get_name() + "]::Running")

    tracker_provider = QlinTrackerProvider(_config['ns_prefix'])

    udp_recv_socket = QUdpSocket()
    recv_addr = QHostAddress.AnyIPv4
    # print(recv_addr.)
    if udp_recv_socket.bind(recv_addr, _config['port']):
        rospy.loginfo(
            "[" + rospy.get_name() + "]:::Listing on UDP Port: "
            + str(_config['port']) + " IP: " + repr(recv_addr)
            + " setup success"
        )
    else:
        rospy.logerr("[" + rospy.get_name() + "]:::Listing on UDP Port: "
                     + str(_config['port']) + " setup failed")
        raise RuntimeError("UDP cannot open")

    try:
        updated_time = rospy.get_time()
        while True:
            if rospy.get_time() - updated_time > timed_out:
                tracker_provider.reset_has_lock()

            if udp_recv_socket.hasPendingDatagrams():
                updated_time = rospy.get_time()

                datagram, host, port = udp_recv_socket.readDatagram(udp_recv_socket.pendingDatagramSize())
                # logger.info(str(datagram))
                data_buffer = QByteArray(datagram)
                crc16 = qChecksum(data_buffer[:-2])
                data_stream = QDataStream(data_buffer, QIODevice.ReadOnly)
                data_stream.setVersion(18)
                data_stream.setByteOrder(QDataStream.BigEndian)

                data = {}
                for key, type, length in incoming_udp_format:
                    tmp_data = []
                    for ind in range(length):
                        if type == 'int':
                            tmp_data.append(data_stream.readUInt32())
                        if type == 'float':
                            tmp_data.append(data_stream.readFloat())
                        if type == 'bool':
                            tmp_data.append(data_stream.readBool())
                    data[key] = tmp_data
                crc_from_sender = data_stream.readUInt16()

                if crc_from_sender != crc16:
                    rospy.loginfo("[" + rospy.get_name() + "]:::CRC Checksum Failled")
                else:
                    tracker_provider.set_tracking_state(has_lock=data['has_lock'][0],
                                                        pos=data['pos'],
                                                        f_length=data['focal_length'])

    except KeyboardInterrupt:
        rospy.loginfo("[" + rospy.get_name() + "]:::exit on keyboard interrupt")

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
