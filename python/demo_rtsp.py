import argparse

import cv2
import numpy
import torch
import rtsp
# from models.with_mobilenet import PoseEstimationWithMobileNet
# from modules.keypoints import extract_keypoints, group_keypoints
# from modules.load_state import load_state
# from modules.pose import Pose, track_poses
# from val import normalize, pad_width


class ImageReader(object):
    def __init__(self, file_names):
        self.file_names = file_names
        self.max_idx = len(file_names)

    def __iter__(self):
        self.idx = 0
        return self

    def __next__(self):
        if self.idx == self.max_idx:
            raise StopIteration
        img = cv2.imread(self.file_names[self.idx], cv2.IMREAD_COLOR)
        if img.size == 0:
            raise IOError('Image {} cannot be read'.format(self.file_names[self.idx]))
        self.idx = self.idx + 1
        return img


class VideoReader(object):
    def __init__(self, file_name):
        self.file_name = file_name
        try:  # OpenCV needs int to read from webcam
            self.file_name = int(file_name)
        except ValueError:
            pass

    def __iter__(self):
        self.cap = cv2.VideoCapture(self.file_name)
        if not self.cap.isOpened():
            raise IOError('Video {} cannot be opened'.format(self.file_name))
        return self

    def __next__(self):
        was_read, img = self.cap.read()
        if not was_read:
            raise StopIteration
        return img





def run_demo(rtscap):
    imgsize=416





    while rtscap.isStarted():
        ok, img = rtscap.read_latest_frame()  # read_latest_frame() 替代 read()
        # if cv2.waitKey(100) & 0xFF == ord('q'):
        #     break
        # if not ok:
        #     continue
        if img is None:
            continue
        orig_img = img.copy()

        resize2fixe=cv2.resize(orig_img,(imgsize, imgsize))


        cv2.imshow('video', resize2fixe)

        if cv2.waitKey(30) == ord('q'):
            break



def cal_correl(pose, humans, filePath):
    with open(filePath, 'r') as fpr:
        dict1 = json.load(fpr)

    list0 = list()
    list1 = list()
    list2 = list()
    list3 = list()

    correl = 0

    # Return correlation 0 while no human detected
    if not len(humans):
        print('No person in sight')
        return 0

    for item in dict1[pose]:
        for key in range(18):

            if key not in humans[0].body_parts.keys() or str(key) not in item.keys():
                continue

            # Add x,y coordinate
            list0.append(item[str(key)][0])
            list1.append(humans[0].body_parts[key].x)
            list2.append(item[str(key)][1])
            list3.append(humans[0].body_parts[key].y)

        # Get value from these two-dimensional arrays
        correlX = numpy.corrcoef(list0, list1)[0][1]
        correlY = numpy.corrcoef(list2, list3)[0][1]
        tmpCorrel = min(abs(correlX), abs(correlY))
        correl = max(tmpCorrel, correl)

        # Reinitiate lists
        list0 = list()
        list1 = list()
        list2 = list()
        list3 = list()

    print(correl)

    return correl


if __name__ == '__main__':
    # parser = argparse.ArgumentParser(
    #     description='''Lightweight human pose estimation python demo.
    #                    This is just for quick results preview.
    #                    Please, consider c++ demo for the best performance.''')
    # parser.add_argument('--checkpoint-path', type=str, default='checkpoint_iter_45000.pth', help='path to the checkpoint')
    # parser.add_argument('--height-size', type=int, default=256, help='network input layer height size')
    # parser.add_argument('--video', type=str, default='', help='path to video file or camera id')
    # parser.add_argument('--images', nargs='+', default='', help='path to input image(s)')
    # parser.add_argument('--cpu', action='store_true', help='run network inference on cpu')
    # parser.add_argument('--track', type=int, default=1, help='track pose id in video')
    # parser.add_argument('--smooth', type=int, default=1, help='smooth pose keypoints')
    # args = parser.parse_args()



    rtscap = rtsp.RTSCapture.create('rtsp://admin:zhuoyu1234@192.168.1.64:554/h264/ch1/main/av_stream')
    rtscap.start_read()  # 启动子线程并改变 read_latest_frame 的指向
    run_demo(rtscap)
    rtscap.stop_read()
    rtscap.release()
    cv2.destroyAllWindows()

