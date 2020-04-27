import os

GT_path = './streelight_GT_DATA'
YOLO_path = './Median_BOUNDING_BOX_resnet34_normal_DATA'

GT_objects = 0
YOLO_objects = 0
IOU = 0
precision = 0
recall = 0
frame_count = 0
accuracy = 0


def get_IOU(boxYOLO, boxGT):
        '''
        Inputs: 
            - boxYOLO: bounding box coordinates (left, top, right, bottom) of YOLO returned bounding box
            - boxGT: bounding box coordinates (left, top, right, bottom) of ground truth bounding box

        Output:
            - iou: Intersection-Over-Union calculation between bounding boxes

        - get_IOU function adapted from https://www.pyimagesearch.com/2016/11/07/intersection-over-union-iou-for-object-detection/
        '''
        boxYOLO = boxYOLO.split(',')
        boxGT = boxGT.split(',')

        # determine the (x, y)-coordinates of the intersection rectangle
        xA = max(int(boxYOLO[0]), int(boxGT[0]))
        yA = max(int(boxYOLO[1]), int(boxGT[1]))
        xB = min(int(boxYOLO[2]), int(boxGT[2]))
        yB = min(int(boxYOLO[3]), int(boxGT[3]))
        # compute the area of intersection rectangle
        interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
        # compute the area of both the prediction and ground-truth rectangles
        boxAArea = (int(boxYOLO[2]) - int(boxYOLO[0]) + 1) * (int(boxYOLO[3]) - int(boxYOLO[1]) + 1)
        boxBArea = (int(boxGT[2]) - int(boxGT[0]) + 1) * (int(boxGT[3]) - int(boxGT[1]) + 1)
        # compute the intersection over union by taking the intersection area and dividing it by the sum of prediction 
        # + ground-truth areas - the interesection area
        iou = interArea / float(boxAArea + boxBArea - interArea)
        # return the intersection over union value
        return iou


for filename in sorted(os.listdir(GT_path)):
        # print('=====================================')
        GT_objects_frame = 0 # hold objects count for this frame
        YOLO_objects_frame = 0 # hold objects count for this frame
        frame_IOU = 0 # total IOU for this frame
        precision_frame = 0
        recall_frame = 0
        frame_accuracy = 0
        frame_count += 1
        object_count = 0

        GT_bb_frame = []
        YOLO_bb_frame = []

        with open(os.path.join(GT_path, filename), 'r', encoding="utf8", errors='ignore') as GT:
            for line_GT in GT:
                if line_GT.startswith("car") or line_GT.startswith("truck"):
                    GT_objects_frame += 1
                if line_GT.startswith("Bounding Box:"):
                    line = line_GT.split(':')
                    line = line[1].split(',')
                    left = line[0]
                    top = line[1]
                    right = line[2]
                    bottom = line[3].split('\n')[0]
                    GT_box = "{},{},{},{}".format(left,top,right,bottom)
                    GT_bb_frame.append(GT_box)
                    # print('GT:',GT_box)

        with open(os.path.join(YOLO_path, filename), 'r', encoding="utf8", errors='ignore') as YOLO:
            for line_YOLO in YOLO:
                if line_YOLO.startswith("car") or line_YOLO.startswith("truck"):
                    YOLO_objects_frame += 1 
                if line_YOLO.startswith("Bounding Box:"):
                    object_count += 1
                    line = line_YOLO.split(':')
                    line = line[1].split(',')
                    left = line[0]
                    top = line[1]
                    right = line[2]
                    bottom = line[3].split('\n')[0]
                    YOLO_box = "{},{},{},{}".format(left,top,right,bottom)
                    YOLO_bb_frame.append(YOLO_box)
                    # print('YOLO:',YOLO_box)
        GT.close()
        YOLO.close()

        for i in range(len(YOLO_bb_frame)):
            IOU_best = 0
            for j in range(len(GT_bb_frame)):
                boxYOLO = YOLO_bb_frame[i]
                boxGT = GT_bb_frame[j]
                iou = get_IOU(boxYOLO, boxGT)
                if (iou > IOU_best):
                    IOU_best = iou
                # print('YOLO:',YOLO_bb_frame[i])
                # print('GT:', GT_bb_frame[j])
            frame_IOU += IOU_best

        # Precision Calculation (tp / (tp + fp))
        tp = GT_objects_frame
        if YOLO_objects_frame > GT_objects_frame:
            fp = YOLO_objects_frame - GT_objects_frame
        else:
            fp = 0
        if(tp + fp) == 0:
            precision_frame = 1
        else:
            precision_frame = tp / (tp + fp)

        # Recall Calculation (tp / (tp + fn))
        tp = GT_objects_frame
        if GT_objects_frame > YOLO_objects_frame:
            fn = GT_objects_frame - YOLO_objects_frame
        else:
            fn = 0
        if (tp + fn) == 0:
            recall_frame = 1
        else:
            recall_frame = tp / (tp + fn)

        # if (object_count > GT_objects_frame):
        #     penalty = 2 * (object_count - YOLO_objects_frame) # introduce penalty for overestimating # objects
        #     YOLO_objects_frame = YOLO_objects_frame - penalty

        # if (YOLO_objects_frame > GT_objects_frame):
        #     penalty = 2 * (YOLO_objects_frame - GT_objects_frame) # introduce penalty for overestimating # objects
        #     YOLO_objects_frame = YOLO_objects_frame - penalty
        
        # calculate accuracy as (tp + tn) / (tp + tn + fp + fn)
        numerator = tp
        denominator = tp + fp + fn
        if denominator == 0:
            frame_accuracy = 0
        else:
            frame_accuracy = numerator / denominator
        accuracy += frame_accuracy
        GT_objects += GT_objects_frame
        YOLO_objects += YOLO_objects_frame
        IOU += frame_IOU
        precision += precision_frame
        recall += recall_frame
       
# accuracy = YOLO_objects / GT_objects
accuracy = accuracy / frame_count
print("Accuracy: {0:0.2f}%".format(accuracy * 100))
totalIOU = IOU / GT_objects
print('IoU: {0:0.3f}'.format(totalIOU))
precision = precision / frame_count
print('Precision: {0:0.3f}'.format(precision))
recall = recall / frame_count
print('Recall: {0:0.3f}'.format(recall))