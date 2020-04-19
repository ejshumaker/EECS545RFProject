import cv2
import numpy as np
import os


def proofOfConcept():
    orig = cv2.imread('data/cheetah_results/cheetah_frm00066_orig.png', cv2.IMREAD_ANYCOLOR)
    img = cv2.imread('data/cheetah_results/cheetah_frm00066.png', cv2.IMREAD_GRAYSCALE)

    cv2.imshow('frame', img)
    cv2.waitKey(0)

    # find contours
    contours, hierarchy = cv2.findContours(img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    for contour in contours:
        x, y, dx, dy = cv2.boundingRect(contour)
        if dx > 10 and dy > 10:
            cv2.rectangle(orig, (x, y), (x + dx, y + dy), (0, 255, 0))

    # create an empty image for contours
    img_contours = orig.copy()

    cv2.imshow('frame', img_contours)
    cv2.waitKey(0)


def nonMaxSuppression(boundingBoxes, inter=10):
    ''' Remove bounding boxes which are encapsulated by others, or within 10 pixels'''
    suppress_list = []
    keep_list = []

    for i in range(len(boundingBoxes)):
        if i in suppress_list:
            continue

        x_i, y_i, dx_i, dy_i = boundingBoxes[i]
        
        for j in range(i + 1, len(boundingBoxes)):
            # check if rect i contains/overlaps with rect j
            x_j, y_j, dx_j, dy_j = boundingBoxes[j]
            
            min_x = min(x_i, x_j)
            max_x = max(x_i + dx_i, x_j + dx_j)
            min_y = min(y_i, y_j)
            max_y = max(y_i + dy_i, y_j + dy_j)

            if (((x_j <= x_i and x_i < x_j + dx_j + inter) or (x_i <= x_j and x_j < x_i + dx_i + inter)) and
                ((y_j <= y_i and y_i < y_j + dy_j + inter) or (y_i <= y_j and y_j < y_i + dy_i + inter))):
                # OVERLAPPING RECTS
                x_i, y_i, dx_i, dy_i = (min_x, min_y, max_x - min_x, max_y - min_y)
                suppress_list.append(j)
        
        keep_list.append((x_i, y_i, dx_i, dy_i))
    
    return keep_list


def multiObjectVideo(dir_name='data/cheetah_results'):
    image_files = []
    for filename in os.listdir(dir_name):
        if filename.endswith(".png"):
            image_files.append(filename)
    image_files = sorted(image_files)

    test_img = cv2.imread(os.path.join(dir_name, image_files[0]))
    # print(test_img.shape)

    # open a videowrite
    codec = cv2.VideoWriter_fourcc(*'MJPG')
    out = cv2.VideoWriter(os.path.join(dir_name, 'multi_objects.avi'), codec, 30.0, (test_img.shape[1], test_img.shape[0]))

    for i in range(len(image_files) // 2):
        # Load Image
        img_name = os.path.join(dir_name, image_files[2 * i + 1])
        image = cv2.imread(img_name)

        # Load Mask
        mask_name = os.path.join(dir_name, image_files[2 * i])
        mask = cv2.imread(mask_name, cv2.IMREAD_GRAYSCALE)

        # find contours
        contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        # remove small blobs
        rects = []
        for contour in contours:
            x, y, dx, dy = cv2.boundingRect(contour)
            if dx > 20 and dy > 20:
                rects.append((x, y, dx, dy))
                cv2.rectangle(image, (x, y), (x + dx, y + dy), (255, 0, 0))

        
        # combine inetersection blobs
        nonOverlappingRects = nonMaxSuppression(rects)

        for rect in nonOverlappingRects:
            x, y, dx, dy = rect
            cv2.rectangle(image, (x, y), (x + dx, y + dy), (0, 255, 0))

        cv2.imshow('image', image)
        cv2.imshow('mask', mask)
        cv2.waitKey()

        out.write(image)
    out.release()


if __name__ == '__main__':
    multiObjectVideo()