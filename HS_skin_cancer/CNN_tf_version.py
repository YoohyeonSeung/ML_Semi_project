import os
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import random
import shutil

from glob import glob
from matplotlib.image import imread
from PIL import Image


source_path = "C:/workspaces/ML_Semi_project/HS_skin_cancer/skin_cancer_data/HAM10000_images_part/"

data = np.loadtxt('./skin_cancer_data/HAM10000_metadata.csv', delimiter=",", dtype=np.string_)

file_name_arr = data[1: , 1]
dx = data[1: , 2]

revised_file_name = []
revised_dx = []
cnt = 0
for f in file_name_arr:
    revised_file_name.append(      str(f)[2:-1]+'.jpg'        )
    revised_dx.append(  str(dx[cnt])[2:-1]                    )
    cnt+=1

test_index = []
# for i in range(15):
#     test_index.append(random.randrange(10015))
#
# for i in range(10015):
#     if i in test_index:
#         filename = revised_file_name[i]
#         src = source_path
#         to_dir = "C:/workspaces/ML_Semi_project/HS_skin_cancer/skin_cancer_data/moved_test_data/" + revised_dx[i] + "/"
#         shutil.move(src + filename, to_dir + filename)
#     else :
#         filename = revised_file_name[i]
#         src = source_path
#         to_dir = "C:/workspaces/ML_Semi_project/HS_skin_cancer/skin_cancer_data/moved_train_data/" + revised_dx[i] + "/"
#         shutil.move(src + filename, to_dir + filename)
#
#
# PIL은 이미지를 load 할 때, numpy는 array

train_data_list = glob('skin_cancer_data_mini/moved_train_data/*/*.jpg')

test_data_list = glob('skin_cancer_data_mini/moved_test_data/*/*.jpg')

# image = np.array(Image.open(path))





# print(train_data[:33]                        )
# print(train_data[34:-17])
# print(train_data[-16:])
#
# print(test_data_list[0])
# test_data = test_data_list[0]
# print(test_data[:32])
# print(test_data[33:-17])
# print(test_data[-16:])
#
def train_path_revise(path):
     return path[:38] + "/"+path[39:-17]+"/"+path[-16:]

for i in range(len(train_data_list)):
    train_data_list[i] = train_path_revise(train_data_list[i])
# \를 /로 수정

def test_path_revise(path):
    return path[:37] + "/"+path[38:-17]+"/"+path[-16:]

for i in range(len(test_data_list)):
    test_data_list[i] = test_path_revise( test_data_list[i] )

def get_label_from_path(path):     # 사진별 Label는 함수(이름값)
    return path.split('/')[-2]


train_label_name_list = []    # train_Label을 담는 리스트
for path in train_data_list:
    train_label_name_list.append(get_label_from_path(path))

test_label_name_list = []    # test_label을 담는 리스트
for path in test_data_list:
    test_label_name_list.append(get_label_from_path(path))


unique_label_names = np.unique(train_label_name_list)   # label 값을 중복허용 안하고 class를 배출

#unique_label_name = ['akiec' 'bcc' 'bkl' 'df' 'mel' 'nv' 'vasc']
def onehot_encode_label(path):
    onehot_label = unique_label_names == get_label_from_path(path)
    # label값을 가진 class에서 각 파일에 해당하는 label과 겹치는 곳을 True로 반환

    onehot_label = onehot_label.astype(np.uint8)
    # true값은 1로 false는 0으로 바꿈
    return onehot_label

path = test_data_list[0]
image = np.array(Image.open(path))
print(image.shape)
#
#
# ## 여기부터 image를 받아서 batch 사이즈 주기(Hyper parameter설정)
# batch_size = 1    # 한번에 처리할 이미지 갯수
# learning_rate = 1e-5
# training_epochs = 1
#
# # 입력받는 이미지 사이즈
# data_height = 450
# data_width = 600
# channel_n = 3    # 흑백의 경우 1, RGB트루 컬러일땐 3
#
# num_classes = 3    # 0~9까지 10개의 클래스
#
#
# tf.set_random_seed(777)
#
# X = tf.placeholder(tf.float32, [None, 450, 600, 3])
# Y = tf.placeholder(tf.float32, [None, 3])
#
# # 첫번째 conv, input_size = [?, 450, 600, 3]
# W1 = tf.Variable(tf.random_normal([3, 3, 3, 8 ], stddev=0.01))
#
# L1 = tf.nn.conv2d(X, W1, strides=[1, 1, 1, 1], padding="SAME")
# L1 = tf.nn.relu(L1)
# L1 = tf.nn.max_pool(L1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")
#
#
# # 두번째 conv, input_size = [?, 225, 300, 8]
# W2 = tf.Variable(tf.random_normal([3, 3, 8, 32], stddev=0.01))
#
# L2 = tf.nn.conv2d(L1, W2, strides=[1, 1, 1, 1], padding="SAME")
# L2 = tf.nn.relu(L2)
# L2 = tf.nn.max_pool(L2, ksize=[1, 3, 3, 1], strides=[1, 3, 3, 1], padding="SAME")
#
#
# # 세번째 conv [?, 75, 100, 32]
# W3 = tf.Variable(tf.random_normal([3, 3, 32, 32], stddev=0.01))
#
# L3 = tf.nn.conv2d(L2, W3, strides=[1, 1, 1, 1], padding="SAME")
# L3 = tf.nn.relu(L3)
#
# # 네번째 conv [?, 75, 100, 32]
# W4 = tf.Variable(tf.random_normal([3, 3, 32, 64], stddev=0.01))
#
# L4 = tf.nn.conv2d(L3, W4, strides=[1, 1, 1, 1], padding="SAME")
# L4 = tf.nn.relu(L4)
# L4 = tf.nn.max_pool(L4, ksize=[1, 3, 4, 1], strides=[1, 3, 4, 1], padding="SAME")
# # 여기까지 완료 후 [?, 25, 25, 64]
#
# #평탄화
# L4_flat = tf.reshape(L4, [-1, 25*25*64])
#
# #shape마지막 10은 클래스 갯수
# W5 = tf.get_variable("W3", shape=[25*25*64, 3], initializer=tf.contrib.layers.xavier_initializer())
# b = tf.Variable(tf.random_normal([3]))
#
# hypothesis = tf.matmul(L4_flat, W5) + b
#
# cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=hypothesis,  labels=Y))
#
# optimizer = tf.train.AdamOptimizer(learning_rate = learning_rate)
# train = optimizer.minimize(cost)
#
#
# with tf.Session() as sess:
#     sess.run(tf.global_variables_initializer())
#
#     for epochs in range(training_epochs):
#         avg_cost = 0
#         total_batch =int( 10000/batch_size)     # training data가 10000개 있음
#
#         ## batch data 만들기
#         batch_image = np.zeros((batch_size, data_height, data_width, channel_n))
#         batch_label = np.zeros((batch_size, num_classes))
#
#
#
#
#
#
#         for i in range(total_batch):     # 100개를 한batch로 하므로 한번에 100번 학습
#
#             for n, path in enumerate(train_data_list[i*batch_size:(i+1)*batch_size]): # batch하는 부분 수정 필수!
#                 image = np.array(Image.open(path))
#                 onehot_label = onehot_encode_label(path)
#                 batch_image[n, :, :, :] = image
#                 batch_label[n, :] = onehot_label
#
#
#             batch_xs = batch_image
#             batch_ys = batch_label
#             feed_dict = {X : batch_xs, Y : batch_ys}
#
#             c, _ =  sess.run([cost, train], feed_dict = feed_dict)
#             avg_cost += c / total_batch
#             print("batch 중 {0}".format(i) )
#
#         print('Epoch : ', "%04d" % (epochs + 1 ), "cost : ", "{:.9f}".format(avg_cost))
#
#     correct_prediction = tf.equal(tf.argmax(hypothesis, 1), tf.argmax(Y, 1))
#     accurancy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
#
#
#     # print('Accurnacy : ', sess.run(accurancy, feed_dict={X:, Y:  }))   test데이터에 labeling이 안되있음
#
#     # test data 선정
#     r = random.randint(0, 15 - 1)
#     test_image = np.zeros((1, data_height, data_width, channel_n))
#     test_image[0, :, :, :] = np.array(test_data_list[r])
#
#     pre_test_index = sess.run(tf.argmax(onehot_encode_label(test_label_name_list[r]), 1))
#
#     true_test_index = pre_test_index.index(1)
#
#
#
#     print("Label : ",  unique_label_names[true_test_index])
#     print("Prediction : ", sess.run(tf.argmax(hypothesis, 1), feed_dict={X :  test_image  }      ))
#
