import numpy as np
import PIL.Image as image
import PIL.ImageDraw as draw

def IOU(box, boxes):
    box_area = (box[2] - box[0]) * (box[3] - box[1])
    area = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    xx1 = np.maximum(box[0], boxes[:, 0])
    yy1 = np.maximum(box[1], boxes[:, 1])
    xx2 = np.minimum(box[2], boxes[:, 2])
    yy2 = np.minimum(box[3], boxes[:, 3])
    #
    w = np.maximum(0, xx2 - xx1)
    h = np.maximum(0, yy2 - yy1)

    inter = w * h
    ovr = np.true_divide(inter, (box_area + area - inter))
    # print('ovr',ovr)
    # print('ovr[0]',ovr[0])

    return ovr[0]

def IOU1(box, boxes):
    box_area = (box[3] - box[1]) * (box[4] - box[2])
    area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 4] - boxes[:, 2])
    xx1 = np.maximum(box[1], boxes[:, 1])
    yy1 = np.maximum(box[2], boxes[:, 2])
    xx2 = np.minimum(box[3], boxes[:, 3])
    yy2 = np.minimum(box[4], boxes[:, 4])
    #
    w = np.maximum(0, xx2 - xx1)
    h = np.maximum(0, yy2 - yy1)

    inter = w * h
    ovr = np.true_divide(inter, (box_area + area - inter))
    # print('ovr',ovr)
    # print('ovr[0]',ovr[0])

    return ovr
def nms(boxes, thresh):
    boxes = boxes.detach().numpy()
    # print(boxes)
    if boxes.shape[0] == 0:
        return np.array([])

    _boxes = boxes[(-boxes[:,0]).argsort()]
    # print(_boxes)
    r_boxes = []

    while _boxes.shape[0] > 1:
        a_box = _boxes[0]
        b_boxes = _boxes[1:]

        r_boxes.append(a_box)
        index = np.where(IOU1(a_box, b_boxes) < thresh)
        _boxes = b_boxes[index]

    if _boxes.shape[0] > 0:
        r_boxes.append(_boxes[0])

    return np.stack(r_boxes)


def convert_to_square(img):
    w,h = img.size
    imgs = image.new(mode='RGB',size = (416,416),color=(128,128,128))

    if w>=h:
        a = int(416*h/w)
        b1=int((416 - a) / 2)
        img = img.resize((416,a))
        imgs.paste(img,(0, b1))
        w1,h1 =img.size
        le = 'b1'
        return imgs, w1, h1, b1,le
    else:
        a =int(416*w/h)
        b2 =int((416 - a) / 2)
        img = img.resize((a,416))
        imgs.paste(img, (b2, 0))
        w1,h1 =img.size
        le = 'b2'
        return imgs,w1,h1,b2,le