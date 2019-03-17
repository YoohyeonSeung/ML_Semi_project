import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import os, os.path

from PIL import Image
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.layers import Flatten
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.preprocessing.image import ImageDataGenerator
from keras import losses

np.random.seed(3)

# dic = {"test":[220, 380], "train":[1275, 3725]}
# classes = ["NORMAL", "PNEUMONIA"]
#
# for dirn in dic:
#     print(dirn)
#     for subdir in classes:
#         print(subdir)
#         path = "C:/workspaces/ML_Semi_project/HS/data/"+dirn+"/"+subdir+"/"
#
#         if(subdir=="NORMAL"):
#             limit_num = dic[dirn][0]
#         else:
#             limit_num = dic[dirn][1]
#         cnt = 1;
#         for f in os.listdir(path):
#             if(cnt<=limit_num):
#                 ext = os.path.splitext(f)    # 파일 이름과 확장자를 list형태로 반환 (이름, 확장자)
#                 upper_ext = ext[1].upper()
#                 if(dirn=="test"):
#                     if (subdir == "NORMAL"):
#                         outfile = os.path.join(path, '/workspaces/ML_Semi_project/HS/data/resized_test',
#                                                classes[0] + str(cnt) + upper_ext)
#                     else:
#                         outfile = os.path.join(path, '/workspaces/ML_Semi_project/HS/data/resized_test',
#                                                classes[1] + str(cnt) + upper_ext)
#                 else:
#                     if (subdir == "NORMAL"):
#                         outfile = os.path.join(path, '/workspaces/ML_Semi_project/HS/data/resized_train/',
#                                                classes[0] + str(cnt) + upper_ext)
#                     else:
#                         outfile = os.path.join(path, '/workspaces/ML_Semi_project/HS/data/resized_train/',
#                                                classes[1] + str(cnt) + upper_ext)
#                 im = Image.open(os.path.join(path, f))
#                 new_img = im.resize((1280, 1280))
#                 new_img.save(outfile)
#                 cnt+=1
#             else:
#                 break

# 데이터 정제 완료


# C:/workspaces/ML_Semi_project/HS/data/resized_test 에 test-image 저장
# C:/workspaces/ML_Semi_project/HS/data/resized_train 에 train-image 저장

train_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    'C:/workspaces/ML_Semi_project/HS/data/resized_train',
    target_size=(1280, 1280),
    batch_size = 10,
    class_mode = 'categorical'
)
test_datagen = ImageDataGenerator(rescale=1./255)

test_generator = test_datagen.flow_from_directory(
        'C:/workspaces/ML_Semi_project/HS/data/resized_test',
        target_size=(1280, 1280),
        batch_size=10,
        class_mode='categorical')

model = Sequential()
model.add(Conv2D(4, kernel_size=(20, 20),
                 strides=4,
                 activation='relu',
                 input_shape=(1280,1280,3)))

model.add(Conv2D(8, (16, 16),
                 strides=2,
                 activation='relu'))

model.add(Conv2D(16, (7, 7),
                 strides=2,
                 activation='relu'))

model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dropout(0.5))
model.add(Dense(128, activation='relu'))
model.add(Dense(2, activation='sigmoid'))

model.compile(loss=losses.mean_squared_error, optimizer='adam', metrics=['accuracy'])


model.fit_generator(
        train_generator,
        steps_per_epoch=10,
        epochs=15)


print("-- Evaluate --")
scores = model.evaluate_generator(test_generator, steps=5)
print("%s: %.2f%%" %(model.metrics_names[1], scores[1]*100))

# 6. 모델 사용하기
print("-- Predict --")
output = model.predict_generator(test_generator, steps=5)
np.set_printoptions(formatter={'float': lambda x: "{0:0.3f}".format(x)})
print(test_generator.class_indices)
print(output)