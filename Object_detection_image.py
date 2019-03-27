# Import packages
import math
import os
import cv2
import numpy as np
import tensorflow as tf
import sys

import linesdetect
from line_reductions import LineReductions as lr


# This is needed since the notebook is stored in the object_detection folder.
sys.path.append("..")

# Import utilites
from utils import label_map_util
from utils import visualization_utils as vis_util
from detection_utils import detection_info as detUtils

# Name of the directory containing the object detection module we're using
MODEL_NAME = 'inference_graph'
IMAGE_NAME = 'test2.jpg'


# Grab path to current working directory
CWD_PATH = os.getcwd()

# Path to frozen detection graph .pb file, which contains the model that is used
# for object detection.
PATH_TO_CKPT = os.path.join(CWD_PATH,MODEL_NAME,'frozen_inference_graph.pb')

# Path to label map file
PATH_TO_LABELS = os.path.join(CWD_PATH,'training','labelmap.pbtxt')

# Path to image
PATH_TO_IMAGE = os.path.join(CWD_PATH,IMAGE_NAME)

# Number of classes the object detector can identify
NUM_CLASSES = 16

# Load the label map.
# Label maps map indices to category names, so that when our convolution
# network predicts `5`, we know that this corresponds to `king`.
# Here we use internal utility functions, but anything that returns a
# dictionary mapping integers to appropriate string labels would be fine
label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
category_index = label_map_util.create_category_index(categories)

# Load the Tensorflow model into memory.
detection_graph = tf.Graph()
with detection_graph.as_default():
    od_graph_def = tf.GraphDef()
    with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
        serialized_graph = fid.read()
        od_graph_def.ParseFromString(serialized_graph)
        tf.import_graph_def(od_graph_def, name='')

    sess = tf.Session(graph=detection_graph)

# Define input and output tensors (i.e. data) for the object detection classifier

# Input tensor is the image
image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')

# Output tensors are the detection boxes, scores, and classes
# Each box represents a part of the image where a particular object was detected
detection_boxes = detection_graph.get_tensor_by_name('detection_boxes:0')

# Each score represents level of confidence for each of the objects.
# The score is shown on the result image, together with the class label.
detection_scores = detection_graph.get_tensor_by_name('detection_scores:0')
detection_classes = detection_graph.get_tensor_by_name('detection_classes:0')

# Number of objects detected
num_detections = detection_graph.get_tensor_by_name('num_detections:0')

# Load image using OpenCV and
# expand image dimensions to have shape: [1, None, None, 3]
# i.e. a single-column array, where each item in the column has the pixel RGB value
image1 = cv2.imread(PATH_TO_IMAGE)
image = cv2.resize(image1,(800,700))
print("Resized image to :",image.shape)
image_expanded = np.expand_dims(image, axis=0)

# Perform the actual detection by running the model with the image as input
(boxes, scores, classes, num) = sess.run(
    [detection_boxes, detection_scores, detection_classes, num_detections],
    feed_dict={image_tensor: image_expanded})

# Draw the results of the detection (aka 'visulaize the results')

vis_util.visualize_boxes_and_labels_on_image_array(
    image,
    np.squeeze(boxes),
    np.squeeze(classes).astype(np.int32),
    np.squeeze(scores),
    category_index,
    use_normalized_coordinates=True,
    line_thickness=8,
    min_score_thresh=0.30)

info_boxes = detUtils.box_info_extractor (
    image,
    np.squeeze (boxes),
    np.squeeze (classes).astype (np.int32),
    np.squeeze (scores),
    category_index,
    use_normalized_coordinates=True,
    line_thickness=8,
    min_score_thresh=0.30
)

result = []
for box in info_boxes:
    result.extend([[box.label,(box.x1,box.y1),(box.x2,box.y2)]])
for boxes in result:
    print(">>>>>>>",boxes)
print("")
print("")
# for items in result:
#      if(items.__getitem__(0) == 'X'):
#          result.remove(items)

# get dimensions of high resolution image
dimensions = image1.shape

# height, width, number of channels in image
height = image1.shape[0]
width = image1.shape[1]
channels = image1.shape[2]

print('Image Dimension    : ', dimensions)
print('Image Height       : ', height)
print('Image Width        : ', width)
print('Number of Channels : ', channels)
 # Resizing Bounding Boxes
y_axis = height/700
x_axis = width/800
print("RATIO H:W",y_axis,x_axis)
for items in result:
    a = items.__getitem__(1).__getitem__(0) * x_axis
    b = items.__getitem__(1).__getitem__(1) * y_axis
    c = items.__getitem__(2).__getitem__(0) * x_axis
    d = items.__getitem__(2).__getitem__(1) * y_axis
    gate = items.__getitem__(0)
    items.clear()
    items.append(gate)
    items.append((a,b))
    items.append((c, d))

    print("***********",items)

print("")
print("Bounding boxes length", len(result))
print("")
print("============================================================================================================================")
print("")
print("")
# All the results have been drawn on image. Now display the image.

cv2.namedWindow('Object detector', cv2.WINDOW_NORMAL)
#cv2.imwrite('low_resolution.jpg',image)
cv2.imshow('Object detector', image)
#####################################################################################################################################

merged_lines_all = lr.merge_lines_logic()

print("Length of merged lines all",len(merged_lines_all))
print("merged_lines_all", merged_lines_all)

img_merged_lines = cv2.imread("test2.jpg")
# count = 0
for line in merged_lines_all:
    ###count = count + 1
    cv2.line(img_merged_lines, (line[0][0], line[0][1]), (line[1][0], line[1][1]), (0, 0, 255), 6)


cv2.namedWindow('Probabilstic_Hough', cv2.WINDOW_NORMAL)
cv2.imshow('Probabilstic_Hough', img_merged_lines)
######################################################################################################################
print("")
print("==================================================================================================================================")
print("")
print("")
print('BOUNDING BOXES', result)
result2 = []
bounding_boxes = []
for boxes in result:
    #print("XXXXXXXXX",boxes)
    e = boxes.__getitem__(0)
    a = boxes.__getitem__(1).__getitem__(0)
    b = boxes.__getitem__(1).__getitem__(1)
    c = boxes.__getitem__(2).__getitem__(0)
    d = boxes.__getitem__(2).__getitem__(1)
    result2.extend([[(a, b), (c, d)]])
    bounding_boxes.extend([[e, (a, b), (c, d)]])


print('BOUNDING BOXES LINES ALL',result2)
print("")
print('BOUNDING BOXES LINES + LABELS ALL',bounding_boxes)
print("")
print('MERGED LINES ALL', merged_lines_all)
print("")
print('MERGED LINES ALL LENGTH', len(merged_lines_all))
print("")

result3 = []
for values in result2:
    topleft_x = values.__getitem__(0).__getitem__(0)
    topleft_y = values.__getitem__(0).__getitem__(1)
    bottomright_x = values.__getitem__(1).__getitem__(0)
    bottomright_y = values.__getitem__(1).__getitem__(1)
    print("Considering bounding box",[(topleft_x,topleft_y),(bottomright_x,bottomright_y)])

    for merged_lines in merged_lines_all:
        point1_x = merged_lines.__getitem__(0).__getitem__(0)
        point1_y = merged_lines.__getitem__(0).__getitem__(1)
        point2_x = merged_lines.__getitem__(1).__getitem__(0)
        point2_y = merged_lines.__getitem__(1).__getitem__(1)

        if((topleft_x < point1_x < bottomright_x) and (topleft_x < point2_x < bottomright_x) and(topleft_y < point1_y < bottomright_y)
                and(topleft_y < point2_y < bottomright_y)):
            result3.extend([[(point1_x,point1_y),(point2_x,point2_y)]])

    print("lines to remove", result3)
    print("***********************************************")
print("Final remove list", result3)
print("Length of Final remove list:", len(result3))




"""Remove inside bounding boxes' merging lines"""
for remove_lines in result3:
    for merged_lines in merged_lines_all:
        if(((merged_lines.__getitem__(0).__getitem__(0)) == (remove_lines.__getitem__(0).__getitem__(0))) and
                ((merged_lines.__getitem__(0).__getitem__(1)) == (remove_lines.__getitem__(0).__getitem__(1))) and
                ((merged_lines.__getitem__(1).__getitem__(0)) == (remove_lines.__getitem__(1).__getitem__(0))) and
                ((merged_lines.__getitem__(1).__getitem__(1)) == (remove_lines.__getitem__(1).__getitem__(1)))):

            merged_lines_all.remove([(merged_lines.__getitem__(0).__getitem__(0),merged_lines.__getitem__(0).__getitem__(1)),(merged_lines.__getitem__(1).__getitem__(0),merged_lines.__getitem__(1).__getitem__(1))])




"""Remove any pair of values from the merging_lines _all array , if the coordinate of any point get 0 value"""
result4 = []
for line in merged_lines_all:
        if((line[0][0] != 0) and (line[0][1] != 0) and (line[1][0] != 0) and (line[1][1] != 0)):

            result4.extend([[(line[0][0],line[0][1]),(line[1][0],line[1][1])]])

img2 = cv2.imread("test2.jpg")
for line in result4:
    print("LINE :", line)
    cv2.line(img2, (line[0][0], line[0][1]), (line[1][0], line[1][1]), (34, 196, 9), 6)

cv2.namedWindow('AFTER REMOVE INSIDE LINES', cv2.WINDOW_NORMAL)
cv2.imshow('AFTER REMOVE INSIDE LINES', img2)
print("")
print("===========================================================================================================================================")
print("")
print("")
print("NEW MERGED LINES ALL",result4)
print("")
print("NEW MERGED LINES ALL LENGTH",len(result4))
print("")
boxes = []
junctions =[]
nodes = []
for items in bounding_boxes:
    if(items.__getitem__(0) == 'T'):
        junctions.extend([items])
    elif(items.__getitem__(0) == 'NODE'):
        nodes.extend([items])
    else:
        boxes.extend([items])
result5 = []
a =1
for lines in result4:
    start_point = lines[0]
    end_point = lines[1]

    for others in range(a, len(result4)):
        point1 = result4[others].__getitem__(0)
        point2 = result4[others].__getitem__(1)

        if((start_point == point1) or (end_point == point2)):
            result5.extend([result4[others]])

    a = a+1
for a in result5:
    print("*****DUPLICATE START OR END POINTS****",a)
"""Remove duplicate start or end points """
for a in result5:
    for lines in result4:
        if(a ==lines):
            result4.remove(a)
print("")
print("")
print('UPDATED MERGED LINES ALL LENGTH',len(result4))

print("")
print("=====================================GROUPING NODES==========================================")
print("")
print("")
new_merged_lines_all = result4



print("")
new_nodes = []
a =1
for nodess in nodes:
    x1 = nodess[1].__getitem__(0)
    y1 = nodess[1].__getitem__(1)
    x2 = nodess[2].__getitem__(0)
    y2 = nodess[2].__getitem__(1)
    #print("NODES",x1,y1,x2,y2)
    for location in range(a,len(nodes)):
        x11 = nodes[location].__getitem__(1).__getitem__(0)
        y11 = nodes[location].__getitem__(1).__getitem__(1)
        x22 = nodes[location].__getitem__(2).__getitem__(0)
        y22 = nodes[location].__getitem__(2).__getitem__(1)
        #print("+++++++++++++", x11, y11, x22, y22)
        if((x1-100 < x11 < x2 +100) and(x1-100 < x22 < x2 +100)  and (y1-300 < y11 < y2+300 ) and(y1-300 < y22 < y2 +300)  ):
            #print("XXXXXXXXX",x11,y11,x22,y22)
            new_nodes.extend([[[(x1,y1),(x2,y2)],[(x11,y11),(x22,y22)]]])
    a = a+1
print("Grouped Nodes array",new_nodes)
print()
print("")
print("")

print("=====================================IDENTIFY THE T-JUNCTION WITH ITS HORIZONTAL LINE======================================================================================================")
print("")


junctions_with_base_line = []

for all_the_lines in new_merged_lines_all:
    x1 = all_the_lines[0].__getitem__(0)
    y1 = all_the_lines[0].__getitem__(1)
    x2 = all_the_lines[1].__getitem__(0)
    y2 = all_the_lines[1].__getitem__(1)
    y11 = y1 - 100
    y22 = y2 + 100
    print("End points of each line segments",x1,y1,x2,y2)
    for t_junctions in junctions:
        a = t_junctions[1].__getitem__(0)
        b = t_junctions[2].__getitem__(0)
        c = t_junctions[1].__getitem__(1)
        d= t_junctions[2].__getitem__(1)
        #print("KKKKKK",t_junctions[2].__getitem__(1))
        if((x1 < a < x2) and (x1 < b < x2) and
                (y11 < c < y22) and (y11 < d < y22)):
            print("")
            print("<<<<<<<<<<<<<<GOT THE END POINT OF LINE WHICH IS INSIDE THE T -JUNCTION>>>>>>>>>>>>>>>")
            print("")
            junctions_with_base_line.extend([[(t_junctions[1].__getitem__(0),t_junctions[1].__getitem__(1)),
                                              (t_junctions[2].__getitem__(0),t_junctions[2].__getitem__(1)),
                                              (x1,y1,x2,y2)]])
print("")
print("")
print("<<<<<<<<<<<T JUNCTIONS+ BASE LINE array >>>>>>>>>>>>>>>")
print(junctions_with_base_line)
print("")
print("")
print("=================================GET THE STARTING POINT OF HORIZONTAL LINE IN T_JUNCTION + OTHER END POINT OF THE LINE SEGMENT IN T_JUNCTION================================================")
print("")
print("")


add_base_point = []
for junc in junctions_with_base_line:
    x1 = junc[0].__getitem__(0)
    y1 = junc[0].__getitem__(1)
    x2 = junc[1].__getitem__(0)
    y2 = junc[1].__getitem__(1)
    a = junc[2].__getitem__(0)
    b = junc[2].__getitem__(1)
    c = junc[2].__getitem__(2)
    d = junc[2].__getitem__(3)

    #print("OOOOO",x1,y1,x2,y2)
    for all_lines in new_merged_lines_all:
        p = all_lines[0].__getitem__(0)
        q = all_lines[0].__getitem__(1)
        r = all_lines[1].__getitem__(0)
        s = all_lines[1].__getitem__(1)
        if(((x1 <p <x2) and (y1 < q< y2))):
          add_base_point.extend([[(a, b), (r, s)]])

        if((x1 < r < x2) and (y1 < s < y2)):
            add_base_point.extend([[(a, b), (p, q)]])

print("ADD_BASE_POINT_ARRAY",add_base_point)
print("")
print("")
print("=====================================REMOVE THE VERTICAL LINE SEGMENT in T_JUNCTION FORM THE TOTAL COUNT OF LINES=================================================================================")
for lines in add_base_point:
    x = lines[1].__getitem__(0)
    y = lines[1].__getitem__(1)
    for alllines in new_merged_lines_all:
        x1 = alllines[0].__getitem__(0)
        y1 = alllines[0].__getitem__(1)
        x2 = alllines[1].__getitem__(0)
        y2 = alllines[1].__getitem__(1)
        if(x ==x1 or y == y1 or x ==x2 or y ==y2):
            new_merged_lines_all.remove(alllines)

print("New number of total lines=",len(new_merged_lines_all))
print("")
print("")

print("=================================GROUPING THE CLOSEST LINES =================================================================================================================================")
print("")
print("")
final_points = []
new_merged_lines_all.sort()
print("SORTED ARRAY OF TOTAL LINES:", new_merged_lines_all)
print("")
print("")

iterations = [1, 2, 3]
# result5 = []
# x =0
for i in iterations:
    for lines in new_merged_lines_all:
        #print("HHHHHHHH", lines)
        point1_1_x = new_merged_lines_all[0].__getitem__(0).__getitem__(0)
        point1_1_y = new_merged_lines_all[0].__getitem__(0).__getitem__(1)
        point2_2_x = new_merged_lines_all[0].__getitem__(1).__getitem__(0)
        point2_2_y = new_merged_lines_all[0].__getitem__(1).__getitem__(1)
        #print("GGGGGGGGG", point1_1_x)
        print("NUMBERS :", point1_1_x, point1_1_y, point2_2_x, point2_2_y)
        result5 = []
        result5.extend([[(point1_1_x, point1_1_y), (point2_2_x, point2_2_y)]])
        #print("KKKKKKKK", len(result5))

        for location in range(1, len(new_merged_lines_all)):
            print(new_merged_lines_all[location])
            point1_x = new_merged_lines_all[location].__getitem__(0).__getitem__(0)
            point1_y = new_merged_lines_all[location].__getitem__(0).__getitem__(1)
            point2_x = new_merged_lines_all[location].__getitem__(1).__getitem__(0)
            point2_y = new_merged_lines_all[location].__getitem__(1).__getitem__(1)
            # print("First VALUES :", point1_x, point1_y, point2_x, point2_y)
            if (len(result5) <= 1):
                one_x = result5[0].__getitem__(0).__getitem__(0)
                one_y = result5[0].__getitem__(0).__getitem__(1)
                two_x = result5[0].__getitem__(1).__getitem__(0)
                two_y = result5[0].__getitem__(1).__getitem__(1)
                if ((((((one_x) - 23) < point1_x < ((one_x) + 23)) or (((one_x) - 23) < point2_x < ((one_x) + 23))) or (
                        (((two_x) - 23) < point1_x < ((two_x) + 23)) or (((two_x) - 23) < point2_x < ((two_x) + 23)))) and (
                        ((((one_y) - 23) < point1_y < ((one_y) + 23)) or (
                                ((one_y) - 23) < point2_y < ((one_y) + 23))) or (
                                (((two_y) - 23) < point1_y < ((two_y) + 23)) or (
                                ((two_y) - 23) < point2_y < ((two_y) + 23))))):
                    #print("first VALUESS :", point1_x, point1_y, point2_x, point2_y)
                    print("True the condition :", [[(point1_x, point1_y), (point2_x, point2_y)]])
                    result5.extend([[(point1_x, point1_y), (point2_x, point2_y)]])
                    result5.sort()
                    print("length:", len(result5), result5)

            # break
            if (len(result5) != 1):
                for lines in result5:
                    one_xx = lines.__getitem__(0).__getitem__(0)
                    one_yy = lines.__getitem__(0).__getitem__(1)
                    two_xx = lines.__getitem__(1).__getitem__(0)
                    two_yy = lines.__getitem__(1).__getitem__(1)
                    for location in range(0, len(new_merged_lines_all)):
                        point1_xx = new_merged_lines_all[location].__getitem__(0).__getitem__(0)
                        point1_yy = new_merged_lines_all[location].__getitem__(0).__getitem__(1)
                        point2_xx = new_merged_lines_all[location].__getitem__(1).__getitem__(0)
                        point2_yy = new_merged_lines_all[location].__getitem__(1).__getitem__(1)
                        if ((((((one_xx) - 23) < point1_xx < ((one_xx) + 20)) or (
                                ((one_xx) - 23) < point2_xx < ((one_xx) + 23))) or
                             ((((two_xx) - 23) < point1_xx < ((two_xx) + 23)) or (
                                     ((two_xx) - 23) < point2_xx < ((two_xx) + 23)))) and
                                (((((one_yy) - 23) < point1_yy < ((one_yy) + 23)) or (
                                        ((one_yy) - 23) < point2_yy < ((one_yy) + 23))) or
                                 ((((two_yy) - 23) < point1_yy < ((two_yy) + 23)) or (
                                         ((two_yy) - 23) < point2_yy < ((two_yy) + 23))))):
                            #print("second VALUES :", point1_xx, point1_yy, point2_xx, point2_yy)
                            print("True the condtion :", [[(point1_xx, point1_yy), (point2_xx, point2_yy)]])
                            if (result5.__contains__([(point1_xx, point1_yy), (point2_xx, point2_yy)])):
                                print("ALREADY HAS")
                            else:
                                result5.extend([[(point1_xx, point1_yy), (point2_xx, point2_yy)]])
                                result5.sort()
                                print("length:", len(result5), result5)

        for new_lines in result5:
            for merged_lines in new_merged_lines_all:
                if (((merged_lines.__getitem__(0).__getitem__(0)) == (new_lines.__getitem__(0).__getitem__(0))) and (
                        (merged_lines.__getitem__(0).__getitem__(1)) == (new_lines.__getitem__(0).__getitem__(1))) and (
                        (merged_lines.__getitem__(1).__getitem__(0)) == (new_lines.__getitem__(1).__getitem__(0))) and (
                        (merged_lines.__getitem__(1).__getitem__(1)) == (new_lines.__getitem__(1).__getitem__(1)))):
                    new_merged_lines_all.remove(new_lines)

        print("*************", new_merged_lines_all)
        print("%%%%%%%%%%%%%%%%%")
        final_points.extend([result5])
        print("FINAL POINTS", final_points)
        # result5[:]

        # x = x+1
# final_points.extend(result5)
# print("NEW VALUES :", result5)
print("LENGTH OF FINAL POINTS :", len(final_points))

for lines in new_merged_lines_all:
    final_points.extend([[lines]])
print("NEW LENGTH OF FINAL POINTS :", len(final_points))
print("NEW FINAL POINTS", final_points)
for lines in final_points:
    print(lines)
print("")
print("")
print("=================================GROUPING THE CLOSEST LINES =================================================================================================================================")
print("")
print("")
print("new_nodes array",new_nodes)
print("add_base_point array",add_base_point)
print("")
print("")
print("=================================PATTERN 01 - VERTICAL LINE OF T_JUNCTIONN IS CONNECTS WITH ANOTHER NORMAL LINE SEGMENT =================================================================================================================================")
print("")
print("")
remove_base_points = []
for points in add_base_point:
    x = points[1].__getitem__(0)
    y = points[1].__getitem__(1)
    for a in final_points:
        for b in a:
            x1 = b[0].__getitem__(0)
            y1 = b[0].__getitem__(1)
            x2 = b[1].__getitem__(0)
            y2 = b[1].__getitem__(1)

            if (((x1 - 30 < x < x1 + 30) and (y1 - 30 < y < y1 + 30)) or (
                    (x2 - 30 < x < x2 + 30) and (y2 - 30 < y < y2 + 30))):
                print("Normal line point",points)
                remove_base_points.extend([points])
                a.extend([[(points[0].__getitem__(0), points[0].__getitem__(1)), (points[1].__getitem__(0), points[1].__getitem__(1))]])
                break

print("REMOVED BASE POINTS",remove_base_points)
for removes in remove_base_points:
    #print("1111111111", removes)
    for items in add_base_point:
        #print("2222222", items)
        if(items == removes):
            print("Remove point",items)
            add_base_point.remove(removes)

print("Updated add_base_points_array", add_base_point)
print("")
print("")
print("=================================PATTERN 02 - VERTICAL LINE OF T_JUNCTIONN IS CONNECTS WITH A NODE ( UPWARD Y _AXIS AND DOWNWARD Y_AXIS)=================================================================================================================================")
print("")
print("")
print("*******************REMOVE THE NODE FROM THE PAIR WHICH HAS THE END POINT OF VERTICAL LINE SEGMENT IN T_JUNCTION***********************")
print("")
print("")
new_item = []
for points in add_base_point:
    x = points[1].__getitem__(0)
    y = points[1].__getitem__(1)
    xx = points[0].__getitem__(0)
    yy = points[0].__getitem__(1)
    for nodes in new_nodes:
            #print("+++++++++", nodes)
            x1 = nodes[0].__getitem__(0).__getitem__(0)
            y1 = nodes[0].__getitem__(0).__getitem__(1)
            x2 = nodes[0].__getitem__(1).__getitem__(0)
            y2 = nodes[0].__getitem__(1).__getitem__(1)
            x11 = nodes[1].__getitem__(0).__getitem__(0)
            y11 = nodes[1].__getitem__(0).__getitem__(1)
            x22 = nodes[1].__getitem__(1).__getitem__(0)
            y22 = nodes[1].__getitem__(1).__getitem__(1)
            #print("::::::::::::::::::;",x1,y1,x2,y2,x11,y11,x22,y22)
            if((x1 < x < x2) and (y1 < y < y2)):
                print("Removing one of First Node in a pair", nodes[0])
                nodes.remove(nodes[0])
                new_item.extend([[(xx,yy),(x11,y11,x22,y22)]])

            if((x11 < x < x22) and(y11 < y < y22)):
                print("Removing one of Second Node in a pair", nodes[1])
                nodes.remove(nodes[1])
                new_item.extend([[(xx, yy),(x1,y1,x2,y2)]])

    #print("====================================",len(new_nodes))
print("Updates new_nodes_array",new_nodes)
print("")
print("")
print("<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<LINKING THE STARTING POINT OF HORIZONATL LINE WITH THE NODE WHICH IS INFORNT OF THE NODE CONNECTED WITH T_JUNCTION>>>>>>>>>>>>>>>>>>>>>>>")
print("")
print("")
print("New_item array",new_item)
print("")
print("")
print("*****************************************************************************************************************************************")
print("")
print("")

for items in new_item:
    x = items[0].__getitem__(0)
    y = items[0].__getitem__(1)
    #print(">>>>>>>>>>>>>>",x,y)
    x1 = items[1].__getitem__(0)
    y1 = items[1].__getitem__(1)
    x2 = items[1].__getitem__(2)
    y2 = items[1].__getitem__(3)
    #print("@@@@@@", x1, y1, x2, y2)
    for lines in final_points:
        for sublines in lines:
            #print("!!!!!!!!!!!!!!!!!!!!", sublines)
            x11 = sublines[0].__getitem__(0)
            y11 = sublines[0].__getitem__(1)
            x22 = sublines[1].__getitem__(0)
            y22 = sublines[1].__getitem__(1)
            #print("%%%%%%%%%", x11, y11, x22, y22)
            if (((x1 < x11 < x2) and (y1 < y11 < y2))):
                print("******************Connects the vertical line of T-junction with Node which is upward of the y-axis****************")
                lines.extend([[(x, y), (x11, y11)]])
                break
            if (((x1 < x22 < x2) and (y1 < y22 < y2))):
                print("******************Connects the vertical line of T-junction with Node which is downward of the y-axis****************")

                lines.extend([[(x, y), (x22, y22)]])
                break
print("")
print("")

print("=====================================PATTERN 03 - ONLY COUPLE OF NODES AND NO T-JUNCTION=============================================================================================================================================================")
print("")


for points in add_base_point:
    x = points[1].__getitem__(0)
    y = points[1].__getitem__(1)
    for nodes in new_nodes:
        if (len(nodes) == 2):
            #print("***************", nodes)
            new_final_points = []
            for subnodes in nodes:
                #print("TTTTTTTTTTT", subnodes)
                x1 = subnodes[0].__getitem__(0)
                y1 = subnodes[0].__getitem__(1)
                x2 = subnodes[1].__getitem__(0)
                y2 = subnodes[1].__getitem__(1)
                #print("@@@@@@", x1, y1, x2, y2)
                for lines in final_points:
                    for sublines in lines:
                        x11 = sublines[0].__getitem__(0)
                        y11 = sublines[0].__getitem__(1)
                        x22 = sublines[1].__getitem__(0)
                        y22 = sublines[1].__getitem__(1)
                        print("@@@@@@", x11, y11, x22, y22)
                        if (((x1 < x11 < x2) and (y1 < y11 < y2)) or ((x1 < x22 < x2) and (y1 < y22 < y2))):
                            final_points.remove(lines)
                            new_final_points.extend([lines])

            #print("RRRRRRRRRRRr", new_final_points)
            total = []
            for xx in new_final_points:
                total += xx
            print(total)
            final_points.extend([total])

print("")
print("")

print("========================================FINAL SETS OF GROUPED LINES==============================================================================================================================================================")
for lines in final_points:
    lines.sort()
    print(">>>>>>>>>>>>>>>>>>>", lines)

#print("HHHHHHHH", final_points)

print("")
print("")
print("========================================TREE STRUCTURE TO GET THE OUTPUT==============================================================================================================================================================")
print("")
print("")

def dist(pt1, pt2):
    return (pt1[0] - pt2[0]) ** 2 + (pt1[1] - pt2[1]) ** 2


def seg_dist(seg1, seg2):
    distances = [dist(seg1[i], seg2[j]) for i in range(2) for j in range(2)]
    return min(enumerate(distances), key=lambda x: x[1])


sorted_lines = []
for lines in final_points:
    connected_part = lines[0]
    non_connected = lines[1:]
    while non_connected:
        mat_dist = [seg_dist(connected_part, non_connected[i])[1] for i in range(len(non_connected))]
        i, min_dist = min(enumerate(mat_dist), key=lambda x: x[1])
        seg_to_connect = non_connected.pop(i)
        idx, real_dist = seg_dist(connected_part, seg_to_connect)
        if idx == 0:
            print("error: this case is not handled")
            exit()
        elif idx == 1:
            print("error: this case is not handled")
            exit()
        elif idx == 2:
            connected_part[1] = seg_to_connect[1]
        elif idx == 3:
            connected_part[1] = seg_to_connect[0]
    sorted_lines.append(connected_part)


class node():
    def __init__(self, name, box) -> None:
        super().__init__()
        self.name = name
        self.box = [(min(box[0][0], box[1][0]), min(box[0][1], box[1][1])),
                    (max(box[0][0], box[1][0]), max(box[0][1], box[1][1]))]
        self.args = []
        self.outputs = []

    def __contains__(self, item):
        return self.box[0][0] <= item[0] <= self.box[1][0] and self.box[0][1] <= item[1] <= self.box[1][1]

    def __str__(self) -> str:
        if self.args:
            return f"{self.name}{self.args}"
        else:
            return f"{self.name}"

    def __repr__(self) -> str:
        return self.__str__()

    def center(self):
        return (self.box[0][0] + self.box[1][0]) / 2, (self.box[0][1] + self.box[1][1]) / 2


nodes = [node(box[0], box[1:]) for box in boxes]

for line in sorted_lines:
    start_point = line[0]
    end_point = line[1]
    try:
        gate1 = next(node for node in nodes if start_point in node)
        gate2 = next(node for node in nodes if end_point in node)
        if gate1.center() < gate2.center():
            source_gate = gate1
            dest_gate = gate2
        else:
            source_gate = gate2
            dest_gate = gate1
        source_gate.outputs.append(dest_gate)
        dest_gate.args.append(source_gate)
    except StopIteration:
        print(f"{start_point} or {end_point} not in any of the boxes")

print("")
print("")
len = next(node for node in nodes if node.name == "OUTPUT").__sizeof__()
print("\t\t\t\t\t\t\t*****************************************************************************************")
print("\t\t\t\t\t\t\t*\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t*")
print("\t\t\t\t\t\t\t*\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t*")
print("\t\t\t\t\t\t\t*\t\t\t\t\t\t\t", next(node for node in nodes if node.name == "OUTPUT"),"\t\t\t\t\t\t\t*")
print("\t\t\t\t\t\t\t*\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t*")
print("\t\t\t\t\t\t\t*\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t*")
print("\t\t\t\t\t\t\t*****************************************************************************************")




cv2.waitKey(0)
# Clean up
cv2.destroyAllWindows()



