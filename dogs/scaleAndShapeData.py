import cv2
from os import listdir
from os.path import isfile,join

dirnames = ['./beagle',"./bulldog","./greatdane"]
curDir = dirnames[2]

allfiles = [f for f in listdir(curDir) if isfile(join(curDir,f))]


for i in allfiles:
    image = cv2.imread(curDir+"/"+i)
    if (image.shape[0]==0 or image.shape[1] ==0):
        print("image has no width or height")
    elif image.shape[0]<image.shape[1]:
        remage = cv2.resize(image,(int(image.shape[1]*256/image.shape[0]),256),interpolation=cv2.INTER_AREA)
        # remage1 = remage[0:remage.shape[0],0:100]
        remage = remage[0:remage.shape[0],int((remage.shape[1]-256)/2):int(((remage.shape[1]-256)/2)+256)]
        # cv2.imwrite(curDir+"Set/"+i,remage)
    else:
        remage = cv2.resize(image,(256,int(image.shape[0]*256/image.shape[1])),interpolation=cv2.INTER_AREA)
        # remage1 = remage[0:remage.shape[0],0:100]
        # cv2.waitKey(0)
        # print(remage.shape)
        remage = remage[int((remage.shape[0]-256)/2):int(((remage.shape[0]-256)/2)+256),0:remage.shape[1]]
        # cv2.imwrite(curDir+"Set/"+i,remage)

