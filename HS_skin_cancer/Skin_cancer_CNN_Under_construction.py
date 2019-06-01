import matplotlib.pyplot as plt
import numpy as np
import os
import shutil
import keras.optimizers as op
import keras.initializers as kini
import keras.backend as K


from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.layers import Flatten
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.preprocessing.image import ImageDataGenerator
from keras import losses



# source_path = "C:/workspaces/ML_Semi_project/HS_skin_cancer/skin_cancer_data/HAM10000_images_part/"
#
# data = np.loadtxt('./skin_cancer_data/HAM10000_metadata.csv', delimiter=",", dtype=np.string_)
#
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
# for i in range(10015):    ## 8:2 비율로 train data, test_data분류
#     if i%5==0:
#         filename = revised_file_name[i]
#         src = source_path
#         to_dir = "C:/workspaces/ML_Semi_project/HS_skin_cancer/skin_cancer_data/moved_train_data/"+revised_dx[i]+"/"
#         shutil.move(src + filename , to_dir + filename)
#     else :
#         filename = revised_file_name[i]
#         src = source_path
#         to_dir = "C:/workspaces/ML_Semi_project/HS_skin_cancer/skin_cancer_data/moved_test_data/" + revised_dx[i] + "/"
#         shutil.move(src + filename, to_dir + filename)

K.clear_session() # 새로운 세션으로 시작

train_datagen = ImageDataGenerator(rescale=1./255)

train_dir = 'C:/workspaces/ML_Semi_project/HS_skin_cancer/skin_cancer_data/moved_train_data'
test_dir =  'C:/workspaces/ML_Semi_project/HS_skin_cancer/skin_cancer_data/moved_test_data'


train_generator = train_datagen.flow_from_directory(    # train
    train_dir ,
    target_size=(600, 450),
    batch_size = 10,
    class_mode = 'categorical',
    shuffle = True
)
test_datagen = ImageDataGenerator(rescale=1./255)

test_generator = test_datagen.flow_from_directory(     # test
        test_dir,
        target_size=(600, 450),
        batch_size=10,
        class_mode='categorical',
        shuffle = False
)

model = Sequential()
model.add(Conv2D(4, kernel_size=(9, 6),
                 strides=1,
                 kernel_initializer=kini.he_normal(seed=None),
                 activation="relu",
                 input_shape=(600, 450, 3))
          )
model.add(Conv2D(4, kernel_size=(9, 6),
                 strides=1,
                 kernel_initializer=kini.he_normal(seed=None),
                 activation="relu")
          )
model.add(MaxPooling2D(pool_size=(2, 2)))
# 1st conv

model.add(Conv2D(6, (7, 3),
                 strides=1,
                 kernel_initializer=kini.he_normal(seed=None),
                 activation="relu"))
model.add(Conv2D(6, (7, 3),
                 strides=1,
                 kernel_initializer=kini.he_normal(seed=None),
                 activation="relu"))
model.add(MaxPooling2D(pool_size=(2, 2)))
# 2nd conv

model.add(Conv2D(8, (5, 3),
                 strides=1,
                 kernel_initializer=kini.he_normal(seed=None),
                 activation="relu"))
model.add(Conv2D(8, (5, 3),
                 strides= 1,
                 kernel_initializer=kini.he_normal(seed=None),
                 activation="relu"),
                 )
model.add(MaxPooling2D(pool_size=(2, 2)))


model.add(Flatten())
model.add(Dropout(0.7))
model.add(Dense(4096,
                 kernel_initializer=kini.he_normal(seed=None),
                 activation="relu"))
model.add(Dense(2048,
                 kernel_initializer=kini.he_normal(seed=None),
                 activation="relu"))
model.add(Dense(1024,
                 kernel_initializer=kini.he_normal(seed=None),
                 activation="relu"))
model.add(Dense(512,
                kernel_initializer=kini.he_normal(seed=None),
                activation="relu"))
model.add(Dense(256,
                kernel_initializer=kini.he_normal(seed=None),
                activation="relu"))
model.add(Dense(128, activation="relu"))
model.add(Dense(7, activation="softmax"))



model.compile(loss=losses.categorical_crossentropy, optimizer=op.Adam(lr=1e-4), metrics=['accuracy'])


history = model.fit_generator(
        train_generator,
        steps_per_epoch=10,
        epochs=40)


print("-- Evaluate --")
scores = model.evaluate_generator(test_generator, steps=5)
print("%s: %.2f%%" %(model.metrics_names[1], scores[1]*100))

# 6. 모델 사용하기
print("-- Predict --")
output = model.predict_generator(test_generator, steps=5)
np.set_printoptions(formatter={'float': lambda x: "{0:0.3f}".format(x)})
print(test_generator.class_indices)
print(output)

plt.figure(1)
plt.plot(history.history['acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train'], loc='upper left')

# summarize history for loss
plt.figure(2)
plt.plot(history.history['loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train'], loc='upper left')
plt.show()


# print(model.summary())








