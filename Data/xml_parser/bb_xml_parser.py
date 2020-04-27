import os

path = '../labels_streelight1/'

file_GT = open("streelight_BB_GT.txt", "w")

for filename in sorted(os.listdir(path)):
    if not filename.endswith('.xml'): continue
    with open(os.path.join(path,filename), 'r') as f:
        file_GT.write('frame\n')
        # print('FPS:')
        file_GT.write('Objects: \n\n')
        # print('Objects: \n')
        for line in f:
            if line.split()[0] == "<object>":
                obj = next(f)
                obj = obj.split('>')
                obj = obj[1].split('<')[0]
                file_GT.write(obj + ':\n')
                # print(obj + ':')

            if line.split()[0] == "<bndbox>":
                xmin = next(f)
                xmin = xmin.split('>')
                xmin = xmin[1].split('<')[0]

                ymin = next(f)
                ymin = ymin.split('>')
                ymin = ymin[1].split('<')[0]

                xmax = next(f)
                xmax = xmax.split('>')
                xmax = xmax[1].split('<')[0]

                ymax = next(f)
                ymax = ymax.split('>')
                ymax = ymax[1].split('<')[0]
                bbox = "Bounding Box:{},{},{},{}".format(xmin,ymin,xmax,ymax)
                file_GT.write(bbox + '\n')
                # print(bbox)
file_GT.close()


