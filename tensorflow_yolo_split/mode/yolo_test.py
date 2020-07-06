import os
import tensorflow as tf
import glob
from tqdm import tqdm
import cv2
import matplotlib.pyplot as plt
#import tensorflow.contrib.eager as tfe
import numpy as np
from mode.annotation import *
from mode.box import *
from mode.darknet53 import *
from mode.generator import *
from mode.weights import *
from mode.yolohead import *
from mode.yolov3 import *
from mode.yololoss import *
from mode.yolo_test import *
import tensorflow as tf
from tensorflow.keras import backend as K
from PIL import Image, ImageFont, ImageDraw
import concurrent.futures
import time


LABELS = ["方標"]
yolo_v3= Yolonet(n_classes=len(LABELS))
yolo_v3.load_weights("model"+'\\weights'+".h5")
COCO_ANCHORS = [10,13, 16,30, 33,23, 30,61, 62,45, 59,119, 116,90, 156,198, 373,326]
imgsize =1024

def list_add(a,b):
    c = []
    for i in range(len(a)):
        c.append(a[i]+b[i])
    return c

def draw_cv2(scale_factor,img_scaled,boxes, labels, probs,class_labels,obj_thresh,desired_size):
    for box, label, prob in zip(boxes, labels, probs):
        label_str = class_labels[label]
        if prob > obj_thresh:
            #print(label_str + ': ' + str(prob*100) + '%')
                
            # Todo: check this code
            if img_scaled.dtype == np.uint8:
                img_scaled = img_scaled.astype(np.int32)
            x1, y1, x2, y2 = (box * scale_factor).astype(np.int32)
            cv2.rectangle(img_scaled, (x1,y1), (x2,y2), (0,255,0), 1)
    return img_scaled
def draw_boxes_local_(image, boxes, labels, probs, class_labels, obj_thresh=0.0, desired_size=None):
    fontPath = r'C:\Windows\Fonts\kaiu.ttf'
    def _set_scale_factor():
        if desired_size:
            img_size = min(image.shape[:2])
            if img_size < desired_size:
                scale_factor = float(desired_size) / img_size
            else:
                scale_factor = 1.0
        else:
            scale_factor = 1.0
        return scale_factor
    scale_factor = _set_scale_factor()
    h, w = image.shape[:2]
    img_scaled = cv2.resize(image, (int(w*scale_factor), int(h*scale_factor)))
    img_scaled=draw_cv2(scale_factor,img_scaled,boxes, labels, probs,class_labels,obj_thresh,desired_size)
    return img_scaled 
def result_yolo_416(img,imgsize=416):
    try1=time.time()
    coordinate=[]

    or_numb=[]
    #imgs.append(img)
    
    boxes, labels, probs = yolo_v3.detect(img, COCO_ANCHORS,imgsize)
    coordinate.append([boxes,probs])
    or_numb.append([len(boxes)])
    image = draw_boxes_local_(img, boxes, labels, probs, class_labels=LABELS, desired_size=400)

    image = np.asarray(image,dtype= np.uint8)
    plt.figure(figsize=(4, 4), dpi=200, tight_layout=True, linewidth=1, edgecolor='r')
    plt.imshow(image)
    plt.show()
    return len(probs)
    
def result_yolo_test(img,imgsize=416):
    try1=time.time()
    coordinate=[]

    or_numb=[]
    #imgs.append(img)
    
    boxes, labels, probs = yolo_v3.detect(img, COCO_ANCHORS,imgsize)
    coordinate.append([boxes,probs])
    or_numb.append([len(boxes)])
    #image = draw_boxes_local_(img, boxes, labels, probs, class_labels=LABELS, desired_size=400)
    #image = np.asarray(image,dtype= np.uint8)
    #plt.figure(figsize=(4, 4), dpi=200, tight_layout=True, linewidth=1, edgecolor='r')
    #plt.imshow(image)
    #plt.show()
    try2=time.time()
    print('{}'.format(try2-try1))
    return coordinate,or_numb
def result_yolo_test_split(img,imgsize=416):
    try1=time.time()
    coordinate_split=[]
    split_numb=[]
    #imgs.append(img)
    boxes, labels, probs = yolo_v3.detect(img, COCO_ANCHORS,imgsize)
    coordinate_split.append([boxes,probs])
    
    #print(boxes, probs)
    #split_numb.append(len(boxes))
    #print(len(boxes))
    #image = draw_boxes_local_(img, boxes, labels, probs, class_labels=LABELS, desired_size=400)
    #image = np.asarray(image,dtype= np.uint8)
    #plt.figure(figsize=(4, 4), dpi=200, tight_layout=True, linewidth=1, edgecolor='r')
    #plt.imshow(image)
    #plt.show()
    try2=time.time()
    print('{}'.format(try2-try1))
    return coordinate_split

def split_method(img):
    img_size=img.shape
    img_size_width=img_size[1]
    img_size_hight=img_size[0]
    img_size_width_CUT=img_size_width//2
    img_size_hight_CUT=img_size_hight//2
    #split--
    img_LT_coordinate = [0,img_size_width_CUT,0,img_size_hight_CUT]
    img_LB_coordinate = [0,img_size_width_CUT,img_size_hight_CUT,img_size_hight_CUT+img_size_hight_CUT]
    img_RT_coordinate = [img_size_width_CUT,img_size_width_CUT+img_size_width_CUT,0,img_size_hight_CUT]
    img_RB_coordinate = [img_size_width_CUT,img_size_width,img_size_hight_CUT,img_size_hight]
    all_img=img_LT_coordinate,img_LB_coordinate,img_RT_coordinate,img_RB_coordinate
    LT,RT,LB,RB= img_LT_coordinate,img_RT_coordinate,img_LB_coordinate,img_RB_coordinate
    crop_img_LT = img[LT[2]:LT[3], LT[0]:LT[1]]
    crop_img_RT = img[RT[2]:RT[3],RT[0]:RT[1] ]
    crop_img_LB = img[LB[2]:LB[3], LB[0]:LB[1]]
    crop_img_RB = img[RB[2]:RB[3], RB[0]:RB[1]]
    return crop_img_LT,crop_img_RT,crop_img_LB,crop_img_RB,img,img_size_width_CUT,img_size_hight_CUT
    
def return_crop(coordinate,or_numb,img_size_width_CUT,img_size_hight_CUT):
    
    continue_list=[]

    for i in range(len(coordinate)):
        [continue_list.append(ii) for ii in zip(coordinate[i][0],coordinate[i][1])]
    list_new=[(['方標','y:(top, bottom)',(int(yolo_ob[0][1]), int(yolo_ob[0][3])),'x:(left, right)',(int(yolo_ob[0][0]), int(yolo_ob[0][2])),\
    'IOU值:',yolo_ob[1]]) for yolo_ob in continue_list]        

    all_yolo = list_new[or_numb[0][0]+or_numb[1][0]+or_numb[2][0]+or_numb[3][0]:or_numb[0][0]+or_numb[1][0]+or_numb[2][0]+or_numb[3][0]+or_numb[4][0]]
    crop_img_LT_list = list_new[0:or_numb[0][0]]
    crop_img_RT_list = list_new[or_numb[0][0]:or_numb[0][0]+or_numb[1][0]]
    crop_img_LB_list = list_new[or_numb[0][0]+or_numb[1][0]:or_numb[0][0]+or_numb[1][0]+or_numb[2][0]]
    crop_img_RB_list = list_new[or_numb[0][0]+or_numb[1][0]+or_numb[2][0]:or_numb[0][0]+or_numb[1][0]+or_numb[2][0]+or_numb[3][0]]


    for number,crop_img_LB_list_number in enumerate (crop_img_LB_list):
        crop_img_LB_list_number[2]= tuple(list_add(crop_img_LB_list_number[2],[img_size_hight_CUT,img_size_hight_CUT]))

    for numberRB,crop_img_RB_list_number in enumerate (crop_img_RB_list):
        crop_img_RB_list_number[2]= tuple(list_add(crop_img_RB_list_number[2],[img_size_hight_CUT,img_size_hight_CUT]))
        crop_img_RB_list_number[4]= tuple(list_add(crop_img_RB_list_number[4],[img_size_width_CUT,img_size_width_CUT]))

    for numberRT,crop_img_RT_list_number in enumerate (crop_img_RT_list):
        crop_img_RT_list_number[4]= tuple(list_add(crop_img_RT_list_number[4],[img_size_width_CUT,img_size_width_CUT])) 
     
    total_coor = crop_img_LT_list+crop_img_LB_list+crop_img_RB_list+crop_img_RT_list+all_yolo
    for i in total_coor:
        i.append(int(i[2][0]+i[4][0]+i[2][1]+i[4][1]))
    return total_coor,all_yolo
    
def find_split(total_coor,img_size_width_CUT,img_size_hight_CUT,img):
    YOLO_ALL_object=[]
    for yolo_ob in total_coor:
        if yolo_ob[2][0] <img_size_hight_CUT and yolo_ob[2][1] >img_size_hight_CUT:
            YOLO_ALL_object.append(yolo_ob)
        elif yolo_ob[4][0] <img_size_width_CUT and yolo_ob[4][1] >img_size_width_CUT:
            YOLO_ALL_object.append(yolo_ob)  
    left_list_range=[]
    right_list_range=[]
    top_list_range=[]
    bottom_list_range=[]
    for YOLO_range in YOLO_ALL_object:
        top_list_range.append(YOLO_range[2][0])
        bottom_list_range.append(YOLO_range[2][1])
        left_list_range.append(YOLO_range[4][0])
        right_list_range.append(YOLO_range[4][1])
    min_left = int(min(left_list_range, default=0))-15
    min_right = int(max(right_list_range, default=0))+15
    min_top = int(min(top_list_range, default=0))-15
    min_bottom = int(max(bottom_list_range, default=0))+15
    crop_img_yolo = img[min_top:min_bottom, min_left:min_right]
    return crop_img_yolo,min_left,min_right,min_top,min_bottom

def return_split_crop(split_list,total_coor,min_left,min_right,min_top,min_bottom,px=0,Confidence=0.2):
    for list_pic_for2 in split_list:
        list_pic_for2[2]= tuple(list_add(list_pic_for2[2],[min_top-px,min_top-px]))
        list_pic_for2[4]= tuple(list_add(list_pic_for2[4],[min_left,min_left]))
    remove_split=[]
    remove_cut_all=[]
    for i in split_list:
        if i[6] == 0:
            remove_split.append(i) 
    for out_del in remove_split:
        split_list.remove(out_del)

    for all_yolo_ob in total_coor:
        if all_yolo_ob[2][0]>= min_top-px and all_yolo_ob[2][1]<= min_bottom+px \
            and all_yolo_ob[4][0]>= min_left and all_yolo_ob[4][1]<= min_right+px or all_yolo_ob[6]<Confidence:
            remove_cut_all.append(all_yolo_ob) 


    for cut_remove_cut_all in remove_cut_all:
        total_coor.remove(cut_remove_cut_all)

    all_list=total_coor+split_list  
    return all_list
    
def customize_NMS(all_list,Anchor_w_h_sub=25,overlap_value=10):
    ALL=[]
    overlap=[]
    del_overlap=[]
    
    for all_ob in all_list:
        if abs((all_ob[2][1]-all_ob[2][0])-(all_ob[4][1]-all_ob[4][0]))<Anchor_w_h_sub:
            ALL.append(all_ob) 

    for i in ALL:
        for ii in ALL:
            if i[7]!=ii[7] and abs(i[7]-ii[7])<overlap_value:
                overlap.append([i,ii])
    for iii in overlap:
        if iii[0][6]>iii[1][6]:
            del_overlap.append(iii[1])
        elif iii[0][6]<iii[1][6]:
            del_overlap.append(iii[0])
    #print(len(del_overlap))
    try:
        for remove__all in del_overlap:
            ALL.remove(remove__all)
    except:
        print("繼續")
    return ALL    
    
    
    