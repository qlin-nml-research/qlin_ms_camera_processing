import cv2
import os

import numpy as np

from inference_step.inference_realtime_dynamicROI import InferencerDROI
from PyQt5.QtNetwork import QUdpSocket, QHostAddress
from PyQt5.QtCore import QByteArray, QLocale, qChecksum, QDataStream, QIODevice, QBuffer, QTime

import logging

# logger init
logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class TransmitDataStruct:
    _fields = [('x_pos', 'float', 1), ('y_pos', 'float', 1), ('has_lock', 'int', 1)]

    def __init__(self):
        for name_, dtype_, len_ in self._fields:
            setattr(self, name_, np.zeros(len_))


def realtime_stream_main_cv2(inference_param, device_id, show_img, debug, port, ip, **kwargs):
    inference_h = InferencerDROI(**inference_param)
    udp_send_sock = QUdpSocket()
    udp_send_addr = QHostAddress(ip)

    cap = cv2.VideoCapture(device_id)

    if cap.isOpened():
        print("device opened")
    else:
        raise RuntimeError("Capture device not opened")

    output_dim = np.array([int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))])
    out_data = TransmitDataStruct()

    try:
        while True:
            ret, frame = cap.read()
            if ret:
                cv2.imshow("current", frame)

                tip_pos = inference_h.process_frame(frame, debug=debug, show_img=show_img)

                if tip_pos is not None:
                    tip_pos = -(tip_pos - output_dim / 2.0)
                    out_data.x_pos = [tip_pos[0]]
                    out_data.y_pos = [tip_pos[1]]
                    out_data.has_lock = [True]
                else:
                    out_data.x_pos = [0]
                    out_data.y_pos = [0]
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
        device_id=_device_id,
        show_img=_show_img,
        debug=_debug,
        port=21039,
        ip="10.198.113.138",
    )
