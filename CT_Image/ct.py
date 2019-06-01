import numpy as np
import pydicom

import glob
import os.path     # 파일 폴더명만 뽑아내기 위해 필요한 라이브러리

from matplotlib import pyplot as plt

source_file = "C:/workspaces/ML_Semi_project/CT_Image/dicom_dir/"

i=0

for file in glob.glob("C:/workspaces/ML_Semi_project/CT_Image/dicom_dir/*.dcm"):

     if i%20==0:
         filename = os.path.basename(file)    # 폴더의 파일 명만 뽑아오는 방법

         dcm = pydicom.read_file(source_file + filename)
         print(dcm)
         print("------------------------------------------------------------------------")

     i=+1

#      plt.imshow(d.pixel_array, cmap = plt.cm.bone)
#      plt.show()
# plt.imsave(file[0:len(file)-4]+".png", d.pixel_array, cmap = plt.cm.bone)
# file_list = []
#
# for file in glob.glob(source_file+'*.png'):
#     filename = os.path.basename(file)     # 폴더에서 파일 명만 뽑아 오는 방법
#
#     file_list.append(source_file+filename)
