import PIL.ImageDraw as draw
import PIL.Image as Image
import nms_iou
import darknet53
import torch
import flag
import torch.nn as nn
import numpy as np
import torchvision.transforms as transforms
import torch
import PIL.ImageFont as Font
import time
font = Font.truetype(font='1.ttf',size=20)


change_tensor = transforms.Compose([
    transforms.ToTensor()
])
class Detector():
    def __init__(self, param=r'F:\yolo\new\param_combine\1400\para_net.pt', iscuda=True):
    # def __init__(self, param=r'F:\yolo\param_combine\200\para_net.pt', iscuda=True):
        self.param = param
        self.iscuda = iscuda
        self.net = darknet53.MainNet()
        self.net.load_state_dict(torch.load(param))
        self.net.eval()

    def detect(self,image,thresh,thresh1):
        # image1 ,w1,h1= nms_iou.convert_to_square(image)
        image_data = change_tensor(image)
        image_data.unsqueeze_(0)
        # if self.iscuda:
        #     self.net.cuda()
        output_13,output_26,output_52 = self.net(image_data)
        index_13,value_13,clf = self.fliter(output_13,thresh1)
        boxes_13 = self.parse(index_13,value_13,flag.ANCHORS_GROUP[13],32,clf)
        # print('classfy13', classify13)
        boxes13 = nms_iou.nms(boxes_13,thresh)
        # print('boxes_13',boxes_13)
        boxes13 = torch.from_numpy(boxes13).float()
        # boxes13 = torch.cat([boxes13,classify13],dim=1)
        # print(boxes13)

        index_26, value_26,clf= self.fliter(output_26, thresh1)
        boxes_26= self.parse(index_26, value_26, flag.ANCHORS_GROUP[26], 16,clf)
        # print('classfy26', classify)
        boxes26 = nms_iou.nms(boxes_26, thresh)
        boxes26 = torch.from_numpy(boxes26).float()
        # print('boxes_26',boxes_26)

        index_52, value_52,clf= self.fliter(output_52, thresh1)
        boxes_52 = self.parse(index_52, value_52, flag.ANCHORS_GROUP[52], 8,clf)
        # print(boxes_52)
        # print('classfy52',classify)
        boxes52 = nms_iou.nms(boxes_52, thresh)
        boxes52 = torch.from_numpy(boxes52).float()
        # print('boxes_52',boxes_52)
        # print(output_52[:,:,0,0])

        return torch.cat([boxes13, boxes26, boxes52],dim=0)

    def fliter(self,output,thresh1):
        output = output.permute(0, 2, 3, 1)
        output = output.view(output.size(0), output.size(1), output.size(2), 3, -1)
        output_conf = torch.sigmoid_(output[...,0])

        mask = output_conf>thresh1

        value =output[mask]

        index = torch.nonzero(mask)
        # print(value)

        return index,value,value[:,0]

    def parse(self,index,value,anchors,t,clf):
        anchors = torch.Tensor(anchors)
        anchors =anchors/t
        a = index[:, 0]
        b = index[:, 3]

        offset_x = torch.sigmoid_(value[:,1])
        offset_y = torch.sigmoid_(value[:, 2])
        #
        #还原到原图，中心点坐标cx，cy
        cy = (index[:,1].float()+offset_y)*t
        cx = (index[:,2].float()+offset_x)*t

        #根据建议框找到实际框
        bw = (anchors[b,0]*torch.exp(value[:,3]))*t
        bh = (anchors[b,1]*torch.exp(value[:,4]))*t

        x1_ = cx - 0.5 * bw
        y1_ = cy - 0.5 * bh
        x2_ = cx + 0.5 * bw
        y2_ = cy + 0.5 * bh
        box_ = torch.stack([clf,x1_,y1_,x2_,y2_],dim=1)
        # print(box_.size())

        # 找到分类坐标索引
        if value.size(0) == 0:
            classify = torch.tensor([])

        else:
            value_class = torch.softmax(value[:, 5:], dim=1)
            # print(torch.argmax(value_class,dim=1))
            # classify = torch.Tensor([[torch.argmax(value_class[0])]])
            classify = torch.argmax(value_class, dim=1)
            classify = classify.unsqueeze(1).float()
            # print(classify)
        # classify = classify.expand(lll.size(0),classify.size(0))
        if box_.size(0)==0:
            return box_
        else:
            # classify = classify.repeat(box_.size(0),1)
            # print(classify)
            box_classify = torch.cat([box_,classify],dim=1)
            # print(box_classify)

            return box_classify


if __name__ == '__main__':

    image_file = r'F:\yolo\new\img\2.jpg'
    img_file = r'F:\yolo\result\cross'
    num =0
    detector = Detector()

    with Image.open(image_file) as im:
        im,w1,h1,b1,le = nms_iou.convert_to_square(im)
        start_time = time.time()
        boxes = detector.detect(im,0.25,0.5)
        # print('classify',classify)
        print(boxes)
        imDraw = draw.ImageDraw(im)
        for box in boxes:
            cx = int(box[1])
            cy = int(box[2])
            w =  int(box[3])
            h =  int(box[4])
            classfy = int(box[5])
            classify_ = flag.COCO_CLASS[classfy]
            imDraw.rectangle((cx, cy, w, h), outline='blue',width=2)
            imDraw.text((cx,cy-20),text=classify_,font=font,fill='blue')

        end_time = time.time()
        t_time = end_time - start_time
        print('t_time:',t_time)
        im.show()
