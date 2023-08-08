import cv2
import os

import numpy as np
import scipy.io as scio

from inference_step.inference_realtime_dynamicROI import InferencerDROI
from PyQt5.QtNetwork import QUdpSocket, QHostAddress
from PyQt5.QtCore import QByteArray, QLocale, qChecksum, QDataStream, QIODevice, QBuffer, QTime

import time
import logging

# logger init
logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

TARGET_DISPLAY_SIZE = [960, 540]


class TransmitDataStruct:
    _fields = [('pos', 'float', 2), ('has_lock', 'int', 1), ('focal_length', 'float', 2)]

    def __init__(self):
        for name_, dtype_, len_ in self._fields:
            setattr(self, name_, np.zeros(len_))


def realtime_stream_main_cv2(inference_param, cam_param, device_id, show_img, debug, port, ip, crop_space, **kwargs):
    inference_h = InferencerDROI(**inference_param)
    udp_send_sock = QUdpSocket()
    udp_send_addr = QHostAddress(ip)

    enable_recording = False
    if 'recording_path' in kwargs:
        enable_recording = True
        recording_path_ = kwargs['recording_path']

    cap = cv2.VideoCapture(device_id)

    # force mac resolution
    if "target_w_resolution" in kwargs:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, kwargs['target_w_resolution'])
    fps = cap.get(cv2.CAP_PROP_FPS)

    if cap.isOpened():
        print("device opened")
    else:
        raise RuntimeError("Capture device not opened")

    output_dim = np.array([int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))])
    print("Current resolution", output_dim)

    sensor_pos_coff = cam_param['sensor_cell_size'][0] * cam_param['native_resolution'][0] / output_dim
    # sensor_pos_coff = sensor_pos_coff / cam_param['focal_length']
    # print(sensor_pos_coff)
    # print(output_dim)

    if enable_recording:
        print("recording at :", recording_path_ + "original.mp4")
        # print((fps))
        fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
        original_img_recorder = cv2.VideoWriter(recording_path_ + "original.mp4", fourcc, 6, tuple(output_dim))
        # stream = original_img_recorder.add_stream('mpeg4', rate=int(fps))  # alibi frame rate
        # stream.width = output_dim[0]
        # stream.height = output_dim[1]
        # stream.pix_fmt = 'yuv420p'
        original_f_time_writer = open(recording_path_ + "original_time.txt", 'w')
    else:
        original_img_recorder = None
        original_f_time_writer = None

    out_data = TransmitDataStruct()

    try:
        frame_time_start = time.time()
        while True:
            ret, frame = cap.read()
            if ret:
                frame_undistorted = cv2.undistort(frame,
                                                  cam_param['intrinsic'],
                                                  cam_param['distort_coff'])

                frame_show = frame_undistorted.copy()
                if enable_recording and original_img_recorder is not None:
                    original_img_recorder.write(frame_show)
                    original_f_time_writer.write('{:.6f}\n'.format(time.time() - frame_time_start))

                tip_pos = inference_h.process_frame(frame_undistorted, debug=debug, show_img=show_img,
                                                    show_img_size=TARGET_DISPLAY_SIZE)

                cv2.rectangle(frame_show, crop_space[0], crop_space[1],
                              (0, 255, 0), 2)
                cv2.imshow("current", cv2.resize(frame_show, TARGET_DISPLAY_SIZE))

                if tip_pos is not None:
                    tip_pos = (tip_pos - output_dim / 2.0) * sensor_pos_coff
                    print(tip_pos)
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
                    out_data.focal_length = cam_param['focal_length'][0]
                    out_data.has_lock = [True]
                else:
                    out_data.pos = [0, 0]
                    out_data.focal_length = cam_param['focal_length'][0]
                    out_data.has_lock = [False]

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
        # Flush stream

    if enable_recording:
        original_img_recorder.release()
        original_f_time_writer.close()
    cap.release()


recording_dir_path = "E:/ExperimentData/MSCameraAutomation/experiment_recording"
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

    # _device_id = 4  # Mooonshot Master PC
    _device_id = 0  # Local PC
    # _device_id = vid_path  # file

    _inference_param = {
        # "model_path": os.path.join(cw_base_path, "model", 'best_model_960.pth'),
        "model_path": os.path.join(cw_base_path, "model", 'best_model.pth'),
        # "network_img_size": [960, 544],
        "network_img_size": [768, 768],
        "inference_param": {
            "CLASSES": ['drill_tip'],
            "ENCODER": "resnet18",
            "ENCODER_WEIGHTS": "imagenet",
            "ACTIVATION": "sigmoid",
            "DEVICE": "cuda",
            "postive_detect_threshold": 70
        },
    }

    resolution = np.array([3840, 2160])
    crop_offset_scale = np.array([0.2, 0.15])
    roi_start = (resolution * crop_offset_scale).astype(np.int32)
    roi_end = (resolution * (1 - crop_offset_scale)).astype(np.int32)

    recording_path = os.path.join(recording_dir_path, "adapt_0804_exp1_vid_")

    realtime_stream_main_cv2(
        inference_param=_inference_param,
        cam_param=cam_param_,
        device_id=_device_id,
        show_img=_show_img,
        debug=_debug,
        port=21039,
        ip="10.198.113.138",
        crop_space=[roi_start, roi_end],
        recording_path=recording_path,
        target_w_resolution=1920,
        # target_w_resolution=3840,
    )
