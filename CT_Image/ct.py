import numpy as np
import pydicom as dicom
import glob

from PIL import Image
from matplotlib import pyplot as plt

source_file = "C:/workspaces/ML_Semi_project/CT_Image/dicom_dir"

for file in glob.glob("C:/workspaces/ML_Semi_project/CT_Image/dicom_dir/*.dcm"):
    d = dicom.read_file(file)
    plt.imsave(file[0:len(file)-4]+".png", d.pixel_array, cmap = plt.cm.bone)

