import cv2
import os

import numpy as np
import scipy.io as scio

from inference_step.inference_realtime_dynamicROI import InferencerDROI
from PyQt5.QtNetwork import QUdpSocket, QHostAddress
from PyQt5.QtCore import QByteArray, QLocale, qChecksum, QDataStream, QIODevice, QBuffer, QTime

import logging

# logger init
logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class TransmitDataStruct:
    _fields = [('pos', 'float', 2), ('has_lock', 'int', 1), ('focal_length', 'float', 2)]

    def __init__(self):
        for name_, dtype_, len_ in self._fields:
            setattr(self, name_, np.zeros(len_))


def realtime_stream_main_cv2(inference_param, cam_param, device_id, show_img, debug, port, ip, **kwargs):
    inference_h = InferencerDROI(**inference_param)
    udp_send_sock = QUdpSocket()
    udp_send_addr = QHostAddress(ip)

    cap = cv2.VideoCapture(device_id)

    if cap.isOpened():
        print("device opened")
    else:
        raise RuntimeError("Capture device not opened")

    output_dim = np.array([int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))])

    sensor_pos_coff = cam_param['sensor_cell_size'] * cam_param['native_resolution'] / output_dim
    # sensor_pos_coff = sensor_pos_coff / cam_param['focal_length']
    print(sensor_pos_coff)
    # print(output_dim)

    out_data = TransmitDataStruct()

    try:
        while True:
            ret, frame = cap.read()
            if ret:
                frame_undistorted = cv2.undistort(frame,
                                                  cam_param['intrinsic'],
                                                  cam_param['distort_coff'])

                cv2.imshow("current", frame_undistorted)

                tip_pos = inference_h.process_frame(frame_undistorted, debug=debug, show_img=show_img)

                if tip_pos is not None:
                    tip_pos = (tip_pos - output_dim / 2.0) * sensor_pos_coff
                    # tip_pose
                    """
                    calculate from center of the sensor, (in meter) offset on the image plane 
                    [---------------------------]
                    [                           ]
                    [           Center          ]
                    [             x --> +x      ] o--> view into the screen with optical center behind
                    [             |             ]  tip projection on sensor
                    [             v +y          ]
                    [___________________________]
                    """
                    out_data.pos = tip_pos
                    out_data.focal_length = cam_param['focal_length']
                    out_data.has_lock = [True]
                else:
                    out_data.pos = [0, 0]
                    out_data.focal_length = cam_param['focal_length']
                    out_data.has_lock = [False]
                # print(tip_pos)

                outArray = QByteArray()
                outBuffer = QBuffer(outArray)
                outBuffer.open(QIODevice.WriteOnly)
                outStream = QDataStream(outBuffer)
                outStream.setVersion(18)
                outStream.setByteOrder(QDataStream.BigEndian)
                for key_, dtype_, length_ in out_data._fields:
                    seg = getattr(out_data, key_)
                    print(key_, dtype_, length_, seg)
                    for ind in range(length_):
                        if dtype_ == 'int':
                            outStream.writeInt32(seg[ind])
                        if dtype_ == 'float':
                            outStream.writeFloat(seg[ind])
                        if type == 'bool':
                            outStream.writeBool(seg[ind])
                outStream.writeUInt16(qChecksum(outArray))
                ret = udp_send_sock.writeDatagram(outArray, udp_send_addr, port)
                if ret < 0:
                    logger.error("::UDP send failure..")

            else:
                break
    except KeyboardInterrupt:
        print("Exit on Interrupt")

    cap.release()


cw_base_path = os.path.abspath(os.path.join(os.getcwd(), "..", ))
if __name__ == '__main__':
    _show_img = True
    _debug = False
    cam_mat_file_path = "../config/cv2_correction_param.mat"
    cam_data = scio.loadmat(cam_mat_file_path)

    intrinsic_mat_ = cam_data['intrinsicMatrix']
    distortion_coeff_ = cam_data['distortionCoefficients']
    focal_length = cam_data['focalLength']
    cell_size_ = cam_data['cellSize']
    sensor_res = cam_data['sensorResolution']
    cam_param_ = {
        "intrinsic": intrinsic_mat_,
        "distort_coff": distortion_coeff_,
        # made up number
        "focal_length": focal_length,
        "object_plane_h": 0.003,
        "object_plane_w": 0.005,
        "sensor_cell_size": cell_size_,
        "native_resolution": sensor_res,
    }

    _device_id = 4  # Mooonshot Master PC
    # _device_id = vid_path  # file

    _inference_param = {
        "model_path": os.path.join(cw_base_path, "model", 'best_model.pth'),
        "network_img_size": [768, 768],
        "inference_param": {
            "CLASSES": ['drill_tip'],
            "ENCODER": "resnet18",
            "ENCODER_WEIGHTS": "imagenet",
            "ACTIVATION": "sigmoid",
            "DEVICE": "cuda",
            "postive_detect_threshold": 40
        },
    }

    realtime_stream_main_cv2(
        inference_param=_inference_param,
        cam_param=cam_param_,
        device_id=_device_id,
        show_img=_show_img,
        debug=_debug,
        port=21039,
        ip="10.198.113.138",
    )
