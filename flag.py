
PICTURE_WIDTH = 416
PICTURE_HEIGHT = 416

CLASS_NUM = 10
CLASS_NUM1 = 8
COCO_CLASS = ['person','bear','glass','zebra','cat','dog','airplane','Bear doll','sheep','giraffe']

ANCHORS_GROUP = {
    13:[[116,90],[156,198],[373,326]],
    26:[[30,61],[62,45],[59,119]],
    52:[[10,13],[16,30],[33,26]]
}

ANCHORS_GROUP_AREA = {
    13:[x*y for x,y in ANCHORS_GROUP[13]],
    26:[x*y for x,y in ANCHORS_GROUP[26]],
    52:[x*y for x,y in ANCHORS_GROUP[52]],
}
