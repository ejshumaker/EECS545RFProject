
with open('BOUNDING_BOX_street.txt', 'r') as f:
    text = ''
    file_num = 0
    for line in f:
        if line.startswith('FPS'):
            file_name = './BB_YOLO_DATA/frame{0:0=3d}'.format(file_num)
            file_write = open(file_name, "w")
            file_write.write(text)
            file_write.close()
            text = ''
            file_num += 1
        else:
            text = text + line

    file_name = './BB_YOLO_DATA/frame{0:0=3d}'.format(file_num)
    file_write = open(file_name, "w")
    file_write.write(text)
    file_write.close()
