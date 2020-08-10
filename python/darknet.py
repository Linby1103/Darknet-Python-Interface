from ctypes import *
import math
import os
import random
import cv2
import numpy as np
import datetime
import rtsp

classes=['AnQuanMao','Ren','FanGuangYi','PuTongYi']
color=[(255,0,0),(0,255,0),(0,0,255)]
def sample(probs):
    s = sum(probs)
    probs = [a/s for a in probs]
    r = random.uniform(0, 1)
    for i in range(len(probs)):
        r = r - probs[i]
        if r <= 0:
            return i
    return len(probs)-1

def c_array(ctype, values):
    arr = (ctype*len(values))()
    arr[:] = values
    return arr

class BOX(Structure):
    _fields_ = [("x", c_float),
                ("y", c_float),
                ("w", c_float),
                ("h", c_float)]

class DETECTION(Structure):
    _fields_ = [("bbox", BOX),
                ("classes", c_int),
                ("prob", POINTER(c_float)),
                ("mask", POINTER(c_float)),
                ("objectness", c_float),
                ("sort_class", c_int)]


class IMAGE(Structure):
    _fields_ = [("w", c_int),
                ("h", c_int),
                ("c", c_int),
                ("data", POINTER(c_float))]

class METADATA(Structure):
    _fields_ = [("classes", c_int),
                ("names", POINTER(c_char_p))]

    

#lib = CDLL("/home/pjreddie/documents/darknet/libdarknet.so", RTLD_GLOBAL)
lib = CDLL("/home/workspace/install/darknet/libdarknet.so", RTLD_GLOBAL)
lib.network_width.argtypes = [c_void_p]
lib.network_width.restype = c_int
lib.network_height.argtypes = [c_void_p]
lib.network_height.restype = c_int

predict = lib.network_predict
predict.argtypes = [c_void_p, POINTER(c_float)]
predict.restype = POINTER(c_float)

set_gpu = lib.cuda_set_device
set_gpu.argtypes = [c_int]

make_image = lib.make_image
make_image.argtypes = [c_int, c_int, c_int]
make_image.restype = IMAGE

get_network_boxes = lib.get_network_boxes
get_network_boxes.argtypes = [c_void_p, c_int, c_int, c_float, c_float, POINTER(c_int), c_int, POINTER(c_int)]
get_network_boxes.restype = POINTER(DETECTION)

make_network_boxes = lib.make_network_boxes
make_network_boxes.argtypes = [c_void_p]
make_network_boxes.restype = POINTER(DETECTION)

free_detections = lib.free_detections
free_detections.argtypes = [POINTER(DETECTION), c_int]

free_ptrs = lib.free_ptrs
free_ptrs.argtypes = [POINTER(c_void_p), c_int]

network_predict = lib.network_predict
network_predict.argtypes = [c_void_p, POINTER(c_float)]

reset_rnn = lib.reset_rnn
reset_rnn.argtypes = [c_void_p]

load_net = lib.load_network
load_net.argtypes = [c_char_p, c_char_p, c_int]
load_net.restype = c_void_p

do_nms_obj = lib.do_nms_obj
do_nms_obj.argtypes = [POINTER(DETECTION), c_int, c_int, c_float]

do_nms_sort = lib.do_nms_sort
do_nms_sort.argtypes = [POINTER(DETECTION), c_int, c_int, c_float]

free_image = lib.free_image
free_image.argtypes = [IMAGE]

letterbox_image = lib.letterbox_image
letterbox_image.argtypes = [IMAGE, c_int, c_int]
letterbox_image.restype = IMAGE

load_meta = lib.get_metadata
lib.get_metadata.argtypes = [c_char_p]
lib.get_metadata.restype = METADATA

load_image = lib.load_image_color
load_image.argtypes = [c_char_p, c_int, c_int]
load_image.restype = IMAGE

rgbgr_image = lib.rgbgr_image
rgbgr_image.argtypes = [IMAGE]

predict_image = lib.network_predict_image
predict_image.argtypes = [c_void_p, IMAGE]
predict_image.restype = POINTER(c_float)

"""
Process video data
"""
ndarray_image = lib.ndarray_to_image
ndarray_image.argtypes = [POINTER(c_ubyte), POINTER(c_long), POINTER(c_long)]
ndarray_image.restype = IMAGE


def nparray_to_image(img):
    data = img.ctypes.data_as(POINTER(c_ubyte))
    image = ndarray_image(data, img.ctypes.shape, img.ctypes.strides)
    return image



def classify(net, meta, im):
    out = predict_image(net, im)
    res = []
    for i in range(meta.classes):
        res.append((meta.names[i], out[i]))
    res = sorted(res, key=lambda x: -x[1])
    return res

def detect(net, meta, image, thresh=.3, hier_thresh=.5, nms=.45):
    im = load_image(image, 0, 0)
    num = c_int(0)
    pnum = pointer(num)
    predict_image(net, im)
    dets = get_network_boxes(net, im.w, im.h, thresh, hier_thresh, None, 0, pnum)
    num = pnum[0]
    if (nms): do_nms_obj(dets, num, meta.classes, nms);

    res = []
    for j in range(num):
        for i in range(meta.classes):
            if dets[j].prob[i] > 0:
                b = dets[j].bbox
                res.append((meta.names[i], dets[j].prob[i], (b.x, b.y, b.w, b.h)))
    res = sorted(res, key=lambda x: -x[1])
    free_image(im)
    free_detections(dets, num)
    return res


def Iou(box1, box2, wh=False):
    if wh == False:
	    xmin1, ymin1, xmax1, ymax1 = box1
	    xmin2, ymin2, xmax2, ymax2 = box2
    else:
        xmin1, ymin1 = int(box1[0]-box1[2]/2.0), int(box1[1]-box1[3]/2.0)
        xmax1, ymax1 = int(box1[0]+box1[2]/2.0), int(box1[1]+box1[3]/2.0)
        xmin2, ymin2 = int(box2[0]-box2[2]/2.0), int(box2[1]-box2[3]/2.0)
        xmax2, ymax2 = int(box2[0]+box2[2]/2.0), int(box2[1]+box2[3]/2.0)
    # 获取矩形框交集对应的左上角和右下角的坐标（intersection）
    tlx = np.max([xmin1, xmin2])
    tly = np.max([ymin1, ymin2])
    brx = np.min([xmax1, xmax2])
    bry = np.min([ymax1, ymax2])
    # 计算两个矩形框面积
    area1 = (xmax1-xmin1) * (ymax1-ymin1)
    area2 = (xmax2-xmin2) * (ymax2-ymin2)
    inter_area=(np.max([0,brx-tlx]))*(np.max(bry-tly)) #计算交集面积

    iou=inter_area/(area1+area2-inter_area+1e-6)#计算交并比

    return iou


def drawbbox(res,image_path):
    """

    :param res: inflence result
    :param image_path: iamge path
    :return:
    """
    if len(res)==0:
        print("Not bbox found!")
        return

    if not os.path.exists(image_path):
        print("% not found!" %image_path )
        return
    image=cv2.imread(image_path)

    for bbox in res:

        claeese=bbox[0].decode()
        conf=bbox[1]
        xmin=int(bbox[2][0]-bbox[2][2]/2)
        ymin=int(bbox[2][1]-bbox[2][3]/2)
        xmax=int(bbox[2][2]/2 + bbox[2][0])
        ymax=int(bbox[2][3]/2 + bbox[2][1])
        cv2.rectangle(image,(xmin,ymin),(xmax,ymax),color=(0,255,0))
        cv2.putText(image, str(claeese), (xmin, ymin-20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
        cv2.putText(image, str(conf), (xmin, ymin -10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0),1)

        save_path='/mnt/share/test/Test_'+os.path.basename(image_path)
    # cv2.imshow("screen_title", image)

        cv2.imwrite(save_path,image)



counter=0

def drawbbox_usb(res,image):
    """

    :param res: inflence result
    :param image_path: iamge path
    :return:
    """
    global counter
    counter+=1
    if len(res)==0:
        print("Not bbox found!")
        return


    person_f=False
    draw_img=image.copy()
    for bbox in res:
        cls = bbox[0].decode()
        if cls=="person":
            person_f=True


        conf = bbox[1]
        xmin = int(bbox[2][0] - bbox[2][2] / 2)
        ymin = int(bbox[2][1] - bbox[2][3] / 2)
        xmax = int(bbox[2][2] / 2 + bbox[2][0])
        ymax = int(bbox[2][3] / 2 + bbox[2][1])


        cv2.rectangle(image, (xmin, ymin), (xmax, ymax), color=color[1])
        cv2.putText(image, str(cls), (xmin, ymin - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color[1], 1)
        cv2.putText(image, str(conf), (xmin, ymin - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color[1],1)




    if person_f  and counter %15==0:
        nowTime = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
        save_path='/mnt/share/20200722-ch/ch3/image/'+str(nowTime)+"_8_7_shiyanshi.jpg"
        cv2.imwrite(save_path,draw_img)

def detectVideo(net, meta, im, thresh=.55, hier_thresh=.3, nms=.45):
    """
    detect video data
    :param net:
    :param meta:
    :param im:
    :param thresh:
    :param hier_thresh:
    :param nms:
    :return:
    """

    num = c_int(0)
    pnum = pointer(num)

    predict_image(net, im)
    dets = get_network_boxes(net, im.w, im.h, thresh, hier_thresh, None, 0, pnum)
    num = pnum[0]
    if (nms): do_nms_obj(dets, num, meta.classes, nms);

    res = []
    for j in range(num):
        for i in range(meta.classes):
            if dets[j].prob[i] > 0:
                b = dets[j].bbox
                res.append((meta.names[i], dets[j].prob[i], (b.x, b.y, b.w, b.h)))
    res = sorted(res, key=lambda x: -x[1])
    free_image(im)
    free_detections(dets, num)
    return res

def Video(base_name):

    cap = cv2.VideoCapture("/mnt/furg-fire-dataset/ele_fire/2020-7-22/VID_20200722_162307.mp4")  # 打开指定路径上的视频文件
    # cap=cv2.VideoCapture(0) #打开设备索引号对于设备的摄像头，一般电脑的默认索引号为0
    net = load_net((base_name + "yolov3.cfg").encode('utf-8'), \
                   (base_name + "backup/yolov3-voc_20000.weights").encode('utf-8'), 0)
    meta = load_meta((base_name + "train.data").encode('utf-8'))

    fps = 16
    size = (640, 480)
    fourcc = cv2.VideoWriter_fourcc('M','J','P','G')
    videoWriter = cv2.VideoWriter('./a.avi', fourcc, fps, size)

    while (True):
        ret, frame = cap.read()

        if ret == True:

            im = nparray_to_image(frame)

            res = detectVideo(net, meta, im)
            drawbbox_usb(res, frame)
            print(res)
            cv2.imshow("screen_title", frame)
            # videoWriter.write(frame)
            if cv2.waitKey(25) & 0xFF == ord('q'):
                break
        else:
            break

    cap.release()
    cv2.destroyAllWindows()

def VideoList(base_name):
    """
    read video in dir
    :return:
    """
    videos=glob.glob(os.path.join("/mnt/share/20200722-ch/ch3/","*.mp4"))
    for video in videos:
        cap = cv2.VideoCapture(video)  # 打开指定路径上的视频文件
        # cap=cv2.VideoCapture(0) #打开设备索引号对于设备的摄像头，一般电脑的默认索引号为0
        print("Load configure from %s"  %(base_name + "yolov3.cfg") )
        # net = load_net((base_name + "yolov3.cfg").encode('utf-8'), \
        #                (base_name + "yolov3.weights").encode('utf-8'), 0)
        # meta = load_meta(("/home/workspace/install/darknet/cfg/coco.data").encode('utf-8'))  #/home/workspace/nnie/package/yolov3/model

        net = load_net((base_name + "cfg/yolov3-voc.cfg").encode('utf-8'), \
                       (base_name + "cfg/backup/yolov3-voc_final.weights").encode('utf-8'), 0)
        meta = load_meta((base_name + "cfg/voc.data").encode('utf-8'))






        fps = 16
        size = (640, 480)
        fourcc = cv2.VideoWriter_fourcc('M', 'J', 'P', 'G')


        while (True):
            ret, frame = cap.read()

            if ret == True:
                h, w = frame.shape[:2]
                cenx, ceny = (w - 1) // 2, (h - 1) // 2

                roi_tl_x=cenx-400
                roi_tl_y=ceny-450
                roi_matrix=frame.copy()
                # roi_matrix=frame[roi_tl_y:roi_tl_y+800,roi_tl_x:roi_tl_x+900,:]
                im = nparray_to_image(roi_matrix)

                # roi_matrix=im.copy()

                # im = nparray_to_image(roi_matrix)

                res = detectVideo(net, meta, im)
                drawbbox_usb(res, roi_matrix)
                print(res)
                cv2.imshow("screen_title", roi_matrix)
                # videoWriter.write(frame)
                if cv2.waitKey(25) & 0xFF == ord('q'):
                    break
            else:
                break

                # for stepSizeH in range(2):
                #     for stepSizeV in range(2):
                #
                #         roi_matrix = frame[stepSizeH * cenx:stepSizeH * cenx + cenx, stepSizeV * ceny:stepSizeV * ceny + ceny, :]
                #
                #         im = nparray_to_image(roi_matrix)
                #
                #         res = detectVideo(net, meta, im)
                #         drawbbox_usb(res, roi_matrix)
                #         print(res)
                #         cv2.imshow("screen_title", roi_matrix)
                #         # videoWriter.write(frame)
                #         if cv2.waitKey(25) & 0xFF == ord('q'):
                #             break
                #     else:
                #         break

    cap.release()
    cv2.destroyAllWindows()






def USBCamera(base_name):
    capture = cv2.VideoCapture(0)
    # /mnt/furg-fire-dataset/barbecue.mp4
    # capture = cv2.VideoCapture("/mnt/furg-fire-dataset/ele_fire/2020-7-22/VID_20200722_162307.mp4")
    net = load_net((base_name+"cfg/yolov3-voc.cfg").encode('utf-8'), \
                   (base_name+"cfg/backup/yolov3-voc.backup").encode('utf-8'), 0)
    meta = load_meta((base_name+"cfg/train.data").encode('utf-8'))
    # 打开自带的摄像头
    if capture.isOpened():
        # 以下两步设置显示屏的宽高
        capture.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        # 持续读取摄像头数据
        while True:
            read_code, frame = capture.read()
            if not read_code:
                break

            im = nparray_to_image(frame)

            res = detectVideo(net, meta,im)
            drawbbox_usb(res, frame)
            print(res)


            cv2.imshow("screen_title", frame)
            # 输入 q 键，保存当前画面为图片
            if cv2.waitKey(30) == ord('q'):
                break
    capture.release()
    cv2.destroyAllWindows()



"""*****************************************************************************************************"""
def Use_Rtsp(rtscap,base_name):
    net = load_net((base_name + "cfg/yolov3-voc.cfg").encode('utf-8'), \
                   (base_name + "cfg/backup/yolov3-voc_final.weights").encode('utf-8'), 0)
    meta = load_meta((base_name + "cfg/voc.data").encode('utf-8'))

    while rtscap.isStarted():
        ok, img = rtscap.read_latest_frame()  # read_latest_frame() 替代 read()

        if img is None:
            continue
        orig_img = img.copy()

        # frame = cv2.resize(orig_img, (416, 416))
        frame=orig_img
        im = nparray_to_image(frame)

        res = detectVideo(net, meta, im)
        drawbbox_usb(res, frame)
        print(res)

        cv2.imshow("screen_title", frame)
        # 输入 q 键，保存当前画面为图片
        if cv2.waitKey(30) == ord('q'):
            break




cam=0
video=0
use_rtsp=1

import glob
if __name__ == "__main__":
    if cam:
        # USBCamera("/mnt/workspcae/caffe/test_model/2020-8-3/") #/mnt/workspcae/darknet/train/train_person/cfg
        USBCamera("/mnt/workspcae/darknet/train/train_person/")
    elif video:
        VideoList("/mnt/workspcae/caffe/test_model/2020-8-3/")
        # VideoList("/home/workspace/nnie/package/yolov3/model/")
        # Video('/mnt/furg-fire-dataset/ele_fire/bangzi/Fire/cfg/')
    elif use_rtsp:
        rtscap = rtsp.RTSCapture.create('rtsp://admin:Aa123456@192.168.1.11:554/Streaming/Channels/301')
        rtscap.start_read()  # 启动子线程并改变 read_latest_frame 的指向
        # Use_Rtsp(rtscap,"/mnt/workspcae/darknet/train/train_person/")  #/mnt/workspcae/caffe/test_model/2020-8-3
        Use_Rtsp(rtscap, "/mnt/workspcae/caffe/test_model/2020-8-3/")

        rtscap.stop_read()
        rtscap.release()
        cv2.destroyAllWindows()
    else:
        # Image list test

        # net = load_net("/mnt/workspcae/caffe/test_model/2020-8-3/cfg/yolov3-voc.cfg".encode('utf-8'), \
        #                "/mnt/workspcae/caffe/test_model/2020-8-3/cfg/backup/yolov3-voc_final.weights".encode('utf-8'), 0)
        # meta = load_meta("/mnt/workspcae/caffe/test_model/2020-8-3/cfg/voc.data".encode('utf-8'))

        net = load_net("/mnt/workspcae/darknet/train/train_tumble/cfg/yolov3-voc.cfg".encode('utf-8'), \
                       "/mnt/workspcae/darknet/train/train_tumble/cfg/backup/yolov3-voc.backup".encode('utf-8'),
                       0)
        meta = load_meta("/mnt/workspcae/darknet/train/train_tumble/cfg/train.data".encode('utf-8'))


        total_counter = 0
        target_counter = 0

        img_list = glob.glob(os.path.join('/media/libin/办公/Workspace/Dataset/tumble/tumble/tumb/', "*.jpg"))
        for img in img_list:
            r = detect(net, meta, img.encode('utf-8'))
            drawbbox(r, img)
            print(r)
            total_counter += 1
            if len(r):
                target_counter += 1

        print("Test image %d pic, found target %d pic" % (total_counter, target_counter))






    

