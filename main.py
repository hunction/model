import os
import matplotlib.pyplot as plt
import glob
import numpy as np
import sympy as sp
import cv2
import torch
from keras.models import load_model

device = "cuda" if torch.cuda.is_available() else "cpu"
print("using {} device".format(device))

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]= "1"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print('Device:', device)
print('Current cuda device:', torch.cuda.current_device())
print('Count of using GPUs:', torch.cuda.device_count())


def padding(img):
    h, w, c = img.shape

    set_size = max(h, w)

    if (h > w):
        delta_w = set_size - w
        delta_h = h - set_size
    elif (h < w):
        delta_w = w - set_size
        delta_h = set_size - h
    elif (h == w):
        return img

    top, bottom = delta_h // 2, delta_h - (delta_h // 2)

    left, right = delta_w // 2, delta_w - (delta_w // 2)

    new_img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=[0, 0, 0])

    return new_img


# test image를 모두 불러와 pre-processing하는 과정
images = sorted(glob.glob('./examples/*.png'))
z = 1
for k in images:
    s = plt.imread(k)
    s = s[:, :, 2] * 255
    s = 255 - s
    for a in range(len(s)):
        for b in range(len(s[0, :])):
            if s[a, b] < 50:
                s[a, b] = 0

    new_s = np.zeros((s.shape[0], s.shape[1] + 10))
    new_s[:, :-10] = s

    i = 0
    j = 0

    for j in range(len(new_s[0, :]) - 1):
        if (new_s[:, j] == 0).all() and (new_s[:, j + 1] > 0).any():

            for i in range(j + 1, len(new_s[0, :]) - 1):
                if (new_s[:, i] > 0).any() and (new_s[:, i + 1] == 0).all():
                    img1 = new_s[:, j:i]
                    cv2.imwrite('./examples/cut_imgs/%d/a%d.png' % (z, i), img1)

    z = z + 1

#cut to slice
y = 0
z = 1
for y in range(6):

    imgs = sorted(glob.glob('./examples/cut_imgs/%d/*' % (y + 1)), key=os.path.getctime)
    z = 1

    for l in imgs:
        org = plt.imread(l)
        org = org * 255
        new = np.zeros((org.shape[0] + 10, org.shape[1]))
        new[5:-5, :] = org

        e = 0
        f = 1

        for f in range(len(new) - 1):
            if (new[f, :] == 0).all() and (new[f + 1, :] > 0).any():
                start = f
                for e in range(len(new) - 1):
                    if (new[-(e + 1), :] == 0).all() and (new[-(e + 2), :] > 0).any():
                        end = len(new) - e
                        break
                break

        org1 = new[start - 1:end + 1, ]
        cv2.imwrite('./examples/slicing_imgs/%d/%d.png' % (y + 1, z), org1)
        z = z + 1

# load_data
test = sorted(glob.glob('./examples/slicing_imgs/4/*.png'), key = os.path.getctime)  # split한 test image들을 불러옴
X = []
for prd in test:
    data = cv2.imread(prd)
    data = padding(data)
    data = data[:, :, 2]
    data = cv2.resize(data, dsize=(28, 28), interpolation=cv2.INTER_CUBIC)
    X.append(data)  # 각각의 image data를 갖는 X배열 생성

X = [cv2.resize(image, (32, 32)) for image in X]  # model에 넣기 위한 data 변환과정
X = np.array(X, dtype="float32")
X = np.expand_dims(X, axis=-1)
X /= 255.0

#load_model
model_all = load_model('Restnet_all.h5')
#model_multi = load_model('Resnet_multi.h5')
#model_A_Z = load_model('Resnet_Big.h5')
#model_mnist = load_model('Resnet_mnist.h5')
#model_a_z = load_model('ResNet_small.h5')
#model_Index = load_model('ResNet_index.h5')
Index = ["/" , "=" , "!" , "Integral" ,"[" , ">=" , ">" , "{" , "(" , "-" , "*","+","]","<=","<","}",")"]

y_hat = model_all.predict(X)  # test image의 target값을 predict
target = []
# target값을 정수형으로 반환
for i in range(len(y_hat)):
    target.append(np.argmax(y_hat[i]))

for j in range(len(target)):
    if target[j] < 10:
        target[j] = "%d" % target[j]

    elif (9 < target[j] and target[j] < 62):
        target[j] = "%c" % (55 + target[j])

    elif target[j] > 61:
        target[j] = Index[target[j] - 62]

    else:
        print("out of bound")

# 0~9 : 숫자
# 10~35 : 대문자 Alphabet A~Z
# 36~61 : 소문자 Alphabet a~z
# 62~78 : 기호  /, =, ! , integral, [ ,>=, > , { , ( , - , * , + , ] , <= , < , } , )
# 총 79개의 Index
print(target)
