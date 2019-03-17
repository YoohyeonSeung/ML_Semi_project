import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
import random
import shutil
import keras.optimizers as op
import math
from glob import glob


from PIL import Image
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.layers import Flatten
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.preprocessing.image import ImageDataGenerator
from keras import losses
from keras.initializers import he_normal


source_path = "C:/workspaces/ML_Semi_project/HS_skin_cancer/skin_cancer_data/HAM10000_images_part/"

data = np.loadtxt('./skin_cancer_data/HAM10000_metadata.csv', delimiter=",", dtype=np.string_)

f = open("C:/workspaces/ML_Semi_project/HS_skin_cancer/result_190313.txt", 'w')

# file_name_arr = data[1: , 1]
# dx = data[1: , 2]
#
# revised_file_name = []
# revised_dx = []
# cnt = 0
# for f in file_name_arr:
#     revised_file_name.append(      str(f)[2:-1]+'.jpg'        )
#     revised_dx.append(  str(dx[cnt])[2:-1]                    )
#     cnt+=1
#
# test_index = []
# for i in range(2000):
#     test_index.append(random.randrange( i*5, (i+1)*5           ))
#
# for i in range(10015):    ## 8:2 비율로 train data, test_data분류
#     if i in test_index:
#         filename = revised_file_name[i]
#         src = source_path
#         to_dir = "C:/workspaces/ML_Semi_project/HS_skin_cancer/skin_cancer_data/moved_test_data/" + revised_dx[i] + "/"
#         shutil.move(src + filename, to_dir + filename)
#     else:
#         filename = revised_file_name[i]
#         src = source_path
#         to_dir = "C:/workspaces/ML_Semi_project/HS_skin_cancer/skin_cancer_data/moved_train_data/"+revised_dx[i]+"/"
#         shutil.move(src + filename , to_dir + filename)

case_cnt = 1
Epoch = 10
for i in range(5):
    for j in range(4):
        train_datagen = ImageDataGenerator(rescale=1./255)
        f.write("{0}번째 시도 \n".format(case_cnt))
        f.write("Epoch = {0}\n".format(Epoch))

        train_generator = train_datagen.flow_from_directory(    # train
            'C:/workspaces/ML_Semi_project/HS_skin_cancer/skin_cancer_data/moved_train_data',
            target_size=(600, 450),
            batch_size = 10,
            class_mode = 'categorical'
        )
        test_datagen = ImageDataGenerator(rescale=1./255)

        test_generator = test_datagen.flow_from_directory(     # test
                'C:/workspaces/ML_Semi_project/HS_skin_cancer/skin_cancer_data/moved_test_data',
                target_size=(600, 450),
                batch_size=10,
                class_mode='categorical')

        model = Sequential()
        model.add(Conv2D(4, kernel_size=(5, 3),
                         strides=1,

                         activation="relu",
                         input_shape=(600, 450, 3)))
        model.add(MaxPooling2D(pool_size=(2, 2)))

        model.add(Conv2D(8, (5, 3),
                         strides=1,

                         activation="relu"))
        model.add(MaxPooling2D(pool_size=(2, 2)))

        model.add(Conv2D(16, (5, 3),
                         strides=2,

                         activation="relu"))

        model.add(Conv2D(32, (3, 2),
                         strides= 2,

                         activation="relu"),
                         )
        model.add(Flatten())
        # model.add(Dropout(0.25))
        model.add(Dense(1024,

                         activation="relu"))
        model.add(Dense(512,

                        activation="relu"))
        model.add(Dense(256,

                        activation="relu"))
        model.add(Dense(128, activation="relu"))
        model.add(Dense(7, activation="softmax"))

        if i == 0 :
            model.compile(loss=losses.categorical_crossentropy, optimizer=op.SGD(math.pow(1/10, j+2)), metrics=['accuracy'])
            a = 'SGD'
            b = '1e-'+str(j+2)
            f.write("Optimzer = "+a + "  , learning_Rate = "+b+"\n")
        elif i == 1 :
            model.compile(loss=losses.categorical_crossentropy, optimizer=op.Adam(math.pow(1/10, j+2)), metrics=['accuracy'])
            a = 'Adam'
            b = '1e-' + str(j + 2)
            f.write("Optimzer = " + a + "  , learning_Rate = " + b+"\n")
        elif i == 2 :
            model.compile(loss=losses.categorical_crossentropy, optimizer=op.RMSprop(math.pow(1/10, j+2)), metrics=['accuracy'])
            a = 'RMSprop'
            b = '1e-' + str(j + 2)
            f.write("Optimzer = " + a + "  , learning_Rate = " + b+"\n")
        elif i == 3 :
            model.compile(loss=losses.categorical_crossentropy, optimizer=op.Adadelta(math.pow(1/10, j+2)), metrics=['accuracy'])
            a = 'Adaedelta'
            b = '1e-' + str(j + 2)
            f.write("Optimzer = " + a + "  , learning_Rate = " + b+"\n")
        elif i == 4 :
            model.compile(loss=losses.categorical_crossentropy, optimizer=op.Adagrad(math.pow(1/10, j+2)), metrics=['accuracy'])
            a = 'Adagrad'
            b = '1e-' + str(j + 2)
            f.write("Optimzer = " + a + "  , learning_Rate = " + b+"\n")

        model.fit_generator(
                train_generator,
                steps_per_epoch=10,
                epochs=Epoch)


        print("-- Evaluate --")
        scores = model.evaluate_generator(test_generator, steps=5)
        print("%s: %.2f%%" %(model.metrics_names[1], scores[1]*100))
        f.write("Evaluate -> Accurancy : " + str(scores[1]*100)+"\n")
        f.write("\n")

        # 6. 모델 사용하기
        #print("-- Predict --")
        output = model.predict_generator(test_generator, steps=5)
        np.set_printoptions(formatter={'float': lambda x: "{0:0.3f}".format(x)})
        #print(test_generator.class_indices)
        #print(output)

f.close()






