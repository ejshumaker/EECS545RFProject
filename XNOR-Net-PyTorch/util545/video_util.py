import cv2
import numpy as np
import os


def video2images(path='../../Data/Streetlight_mask/Morph_Output.avi'):
    cap = cv2.VideoCapture(path)

    file_name = path.split('/')[-1].split('.')[0]
    folder_name = path.split('/')[-2]
    try:
        os.mkdir(os.path.join('../../fastMCD/test', folder_name + '_' + file_name))
    except:
        print("whoops")

    i = 1
    while(cap.isOpened()):
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        cv2.imwrite(os.path.join('../../fastMCD/test', folder_name + '_' + file_name, 'streetlight_frm' + str(i).zfill(5) + '.png'), gray)

        cv2.imshow('frame', gray)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        i += 1

    cap.release()
    cv2.destroyAllWindows()


def images2video(path='../../fastMCD/test/streetlight1'):
    files = os.listdir(path)
    files = sorted(files)

    test_img = cv2.imread(os.path.join(path, files[0]))
    codec = cv2.VideoWriter_fourcc(*'MJPG')
    cap = cv2.VideoWriter('../../fastMCD/test/streelight1.avi', codec, 30.0, (test_img.shape[1], test_img.shape[0]))

    for file in files:
        img = cv2.imread(os.path.join(path, file))
        cap.write(img)
    cap.release()


if __name__ == '__main__':
    images2video()