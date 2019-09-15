import torch
import math
import PIL.ImageDraw as draw
import os
import numpy as np
import torchvision.transforms as transforms
import PIL.Image as image
import torch.utils.data as data
import flag
from nms_iou import IOU,convert_to_square

img_path = r'E:\yolo\img'
label_path = r'E:\yolo\label.txt'

change_tensor = transforms.Compose([
    transforms.ToTensor()
]
)

class MyDataset(data.Dataset):

    def __init__(self):
        with open(label_path) as f:
            self.dataset = f.readlines()


    def __len__(self):

        return len(self.dataset)


    def __getitem__(self, index):
        label = {}
        label_data = self.dataset[index]
        label_data = label_data.split()

        img = image.open(os.path.join(img_path,label_data[0]))
        # print(w_origan,h_origan)
        # img1=img.resize((416,416))

        img,w1,h1,b,le=convert_to_square(img)
        # img.show()
        img_data =change_tensor(img)

        #标签转成浮点型
        label_data1 = np.array([float(x) for x in label_data[1:]])
        #运用np.split直接对列表进行拆分，拆分后的数据类型为array且包含于一个列表
        label_data2 = np.split(label_data1,len(label_data1)//5)

        for feature_size,anchors in flag.ANCHORS_GROUP.items():
            label[feature_size] = np.zeros([feature_size,feature_size,3,5+flag.CLASS_NUM])

            for labl in label_data2:
                conf,cx,cy,w,h = labl
                #标签框
                cx_, w_ = cx*w1,w*w1
                cy_, h_ = cy * h1,h*h1

                # print( cx_offset,cx_index,cy_offset,cy_index)
                if le=='b1':
                    label_x1 = cx_ - 0.5 * w_
                    label_y1 = cy_ - 0.5 * h_+b
                    label_x2 = cx_ + 0.5 * w_
                    label_y2 = cy_ + 0.5 * h_+b
                    label_box = np.array([[label_x1, label_y1, label_x2, label_y2]])
                    cy_=cy_+b
                else:
                    label_x1 = cx_ - 0.5 * w_+b
                    label_y1 = cy_ - 0.5 * h_
                    label_x2 = cx_ + 0.5 * w_+b
                    label_y2 = cy_ + 0.5 * h_
                    label_box = np.array([[label_x1, label_y1, label_x2, label_y2]])
                    cx_=cx_+b

                for i,anchor in enumerate(anchors):
                    #建议框
                    w_anchor = anchor[0]
                    h_anchor = anchor[1]

                    anchor_x1 = cx_ - 0.5 * w_anchor
                    anchor_y1= cy_ - 0.5 * h_anchor
                    anchor_x2 = cx_ + 0.5 * w_anchor
                    anchor_y2 = cy_+ 0.5 * h_anchor
                    anchor_box = np.array([anchor_x1,anchor_y1,anchor_x2,anchor_y2])

                    iou = IOU(anchor_box,label_box)
                    tw = w_/w_anchor
                    th = h_/h_anchor
                    # 特征图上的偏移量和索引
                    cx_offset, cx_index = math.modf(cx_ * feature_size / flag.PICTURE_WIDTH)
                    cy_offset, cy_index = math.modf(cy_ * feature_size / flag.PICTURE_HEIGHT)

                    label[feature_size][int(cy_index),int(cx_index),i]=np.array(
                        [iou,cx_offset,cy_offset,np.log(tw),np.log(th),*onehot(flag.CLASS_NUM,int(conf))])
                    # print('dataset',label[feature_size][int(cy_index),int(cx_index),i])

        return label[13],label[26],label[52],img_data

def onehot(num,ver):
    arr  =np.zeros([num])
    arr[ver]=1
    return arr

mydataset = MyDataset()
mydataset.__getitem__(1)