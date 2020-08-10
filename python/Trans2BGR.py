

import os
import cv2
from numpy import *
import numpy as np

import glob
class JPG2BGR_Solver(object):
    def __init__(self):
        self.img_size = 416  # save bgr size

        self.imgpath = '/mnt/share/shiyanshi/'
        self.path = '/mnt/share/test/2020-07-24-16-05-22_100_cam.bgr'
        # self.path='/home/workspace/nnie/package/HiSVP_PC_V1.1.3.0/software/data/detection/yolov3/000110_416x416.bgr'



    """海思nnie模型需要输入bgr 格式的图片，这个python脚本可以把jpg格式的图片转换成.bgr格式的图片"""

    def jpg2bgr(self):
        save_img_size = self.img_size
        imgpath = self.imgpath
        imgfile=glob.glob(os.path.join(imgpath,"*.jpg"))
        for jpg in imgfile:
            img = cv2.imread(jpg)

            if img is None:
                print("img is none")
            else:
                img = cv2.resize(img, (save_img_size, save_img_size))
                cv2.imwrite(jpg,img)
                (B, G, R) = cv2.split(img)

                savepath=self.imgpath+os.path.basename(jpg)[:-3]+'bgr'
                with open(savepath, 'wb')as fp:
                    for i in range(save_img_size):
                        for j in range(save_img_size):
                            fp.write(B[i, j])
                            print(B[i, j])
                    for i in range(save_img_size):
                        for j in range(save_img_size):
                            fp.write(G[i, j])
                    for i in range(save_img_size):
                        for j in range(save_img_size):
                            fp.write(R[i, j])

                print("save success")


    """查看bgr文件内容并显示为图片"""


    def test_Hi_bgr(self):

        jpeg_path = "/mnt/furg-fire-dataset/ele_fire/Test_fire/fire_1.jpg"
        path = self.path
        imgsize = self.img_size

        f = open(path, 'rb')
        src = cv2.imread(jpeg_path)

        src=np.zeros(src.shape, src.dtype)

        src = cv2.resize(src, (imgsize, imgsize))


        print(src.shape)
        h = src.shape[0]
        w = src.shape[1]
        c = src.shape[2]
        print(f.name)
        (B, G, R) = cv2.split(src)

        data = f.read(imgsize * imgsize * 3)
        for j in range(imgsize):
            for i in range(imgsize):
                B[j, i] = data[j * imgsize + i]
                G[j, i] = data[j * imgsize + i + imgsize * imgsize]
                R[j, i] = data[j * imgsize + i + imgsize * imgsize * 2]

                # B[j, i] = data[j * imgsize + i]
                # G[j, i] = data[j * imgsize + i +1 ]
                # R[j, i] = data[j * imgsize + i +2 ]

        newimg = cv2.merge([R,G,B])



        # cv2.imshow("new", newimg)
        cv2.imwrite('./Test.jpg',newimg)

        f.close()
        cv2.waitKey(0)


if __name__ == '__main__':
    converbgr = True
    solverObj = JPG2BGR_Solver()
    if (converbgr == True):
        solverObj.jpg2bgr()
    else:
        solverObj.test_Hi_bgr()
