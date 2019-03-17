import os
import tensorflow as tf
import matplotlib.pyplot as plt
import random
from glob import glob
from matplotlib.image import imread


# PIL은 이미지를 load 할 때, numpy는 array
from PIL import Image
import numpy as np

data_list = glob('data/trainingSet/*/*.jpg')

test_data_list = glob('data/testSet/*.jpg')


def dir_revising(path):
     return path[:16] + "/"+path[17]+"/"+path[19:]

for i in range(len(data_list)):
    data_list[i] = dir_revising(data_list[i])
# \를 /로 수정

def test_dir_revise(path):
    return path[:12]+"/"+path[13:]

for i in range(len(test_data_list)):
    test_data_list[i] = test_dir_revise( test_data_list[i] )






def get_label_from_path(path):     # 사진별 Label는 함수(이름값)
    return int(path.split('/')[-2])

# rand_n = 9999
# path = data_list[rand_n]
# print(path, get_label_from_path(path))

##  입력받은 이미지가 2차원 배열(28 by 28)인데
##  3차원 배열로 바꿔줌(28 by 28 by 1 )
##  그러므로 RGB트루컬로로 받으면 이과정을 거칠 필요 없음.
def read_image(path):
    image = np.array(Image.open(path))
    return image.reshape(image.shape[0], image.shape[1], 1)




label_name_list = []    # Label를 담는 리스트
for path in data_list:
    label_name_list.append(get_label_from_path(path))

unique_label_names = np.unique(label_name_list)   # label 값을 중복허용 안하고 class를 배출

def onehot_encode_label(path):
    onehot_label = unique_label_names == get_label_from_path(path)
    # label값을 가진 class에서 각 파일에 해당하는 label과 겹치는 곳을 True로 반환

    onehot_label = onehot_label.astype(np.uint8)
    # true값은 1로 false는 0으로 바꿈

    return onehot_label

# print(onehot_encode_label(path))


## 여기부터 image를 받아서 batch 사이즈 주기(Hyper parameter설정)
batch_size = 100    # 한번에 처리할 이미지 갯수
learning_rate = 1e-5
training_epochs = 10

# 입력받는 이미지 사이즈
data_height = 28
data_width = 28
channel_n = 1    # 흑백의 경우 1, RGB트루 컬러일땐 3

num_classes = 10    # 0~9까지 10개의 클래스


# test_n = 10     # batch_size가 100이므로 이것보다 작아야 한다.
# plt.title(batch_label[test_n])
# plt.imshow(batch_image[test_n, :, :, 0])
# plt.show()



tf.set_random_seed(777)

X = tf.placeholder(tf.float32, [None, 28, 28, 1])
Y = tf.placeholder(tf.float32, [None, 10])

# 첫번째 conv, input_size = [?, 28, 28, 1]
W1 = tf.Variable(tf.random_normal([3, 3, 1, 32 ], stddev=0.01))

L1 = tf.nn.conv2d(X, W1, strides=[1, 1, 1, 1], padding="SAME")
L1 = tf.nn.relu(L1)
L1 = tf.nn.max_pool(L1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")


# 두번째 conv, input_size = [?, 14, 14, 32]
W2 = tf.Variable(tf.random_normal([3, 3, 32, 64], stddev=0.01))

L2 = tf.nn.conv2d(L1, W2, strides=[1, 1, 1, 1], padding="SAME")
L2 = tf.nn.relu(L2)
L2 = tf.nn.max_pool(L2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")
# 이과정까지 끝나면 [?, 7, 7, 64]가됨

#평탄화
L2_flat = tf.reshape(L2, [-1, 7*7*64])

#shape마지막 10은 클래스 갯수
W3 = tf.get_variable("W3", shape=[7*7*64, 10], initializer=tf.contrib.layers.xavier_initializer())
b = tf.Variable(tf.random_normal([10]))

hypothesis = tf.matmul(L2_flat, W3) + b

cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=hypothesis,  labels=Y))

optimizer = tf.train.AdamOptimizer(learning_rate = learning_rate)
train = optimizer.minimize(cost)


with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    for epochs in range(training_epochs):
        avg_cost = 0
        total_batch =int( 42000/batch_size)     # training data가 42000개 있음

        ## batch data 만들기
        batch_image = np.zeros((batch_size, data_height, data_width, channel_n))
        batch_label = np.zeros((batch_size, num_classes))






        for i in range(total_batch):     # 100개를 한batch로 하므로 한번에 420번 학습

            for n, path in enumerate(data_list[i*batch_size:(i+1)*batch_size]):
                image = read_image(path)
                onehot_label = onehot_encode_label(path)
                batch_image[n, :, :, :] = image
                batch_label[n, :] = onehot_label


            batch_xs = batch_image
            batch_ys = batch_label
            feed_dict = {X : batch_xs, Y : batch_ys}

            c, _ =  sess.run([cost, train], feed_dict = feed_dict)
            avg_cost += c / total_batch

        print('Epoch : ', "%04d" % (epochs + 1 ), "cost : ", "{:.9f}".format(avg_cost))

    correct_prediction = tf.equal(tf.argmax(hypothesis, 1), tf.argmax(Y, 1))
    accurancy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))


    # print('Accurnacy : ', sess.run(accurancy, feed_dict={X:, Y:  }))   test데이터에 labeling이 안되있음

    # test data 선정
    r = random.randint(0, 28000 - 1)
    test_image = np.zeros((1, data_height, data_width, channel_n))
    test_image[0, :, :, :] = read_image(test_data_list[r])

    To_show_img = np.array(Image.open(test_data_list[r]))

    print("Prediction : ", sess.run(tf.argmax(hypothesis, 1), feed_dict={X :  test_image  }      ))
    plt.imshow(To_show_img)
    plt.show()







