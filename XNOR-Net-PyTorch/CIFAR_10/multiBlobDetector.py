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


def multiObjectVideo(dir_name='data/cheetah_results'):
    image_files = []
    for filename in os.listdir(dir_name):
        if filename.endswith(".png"):
            image_files.append(filename)
    image_files = sorted(image_files)

    test_img = cv2.imread(os.path.join(dir_name, image_files[0]))
    print(test_img.shape)

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

        for contour in contours:
            x, y, dx, dy = cv2.boundingRect(contour)
            if dx > 10 and dy > 10:
                cv2.rectangle(image, (x, y), (x + dx, y + dy), (0, 255, 0))

        cv2.imshow('frame', image)
        cv2.waitKey(10)

        out.write(image)
    out.release()


if __name__ == '__main__':
    multiObjectVideo()