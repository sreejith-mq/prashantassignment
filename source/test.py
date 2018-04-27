import cv2                 # working with, mainly resizing, images
import numpy as np         # dealing with arrays
import os                  # dealing with directories
from random import shuffle # mixing up or currently ordered data that might lead our network astray in training.
from tqdm import tqdm      # a nice pretty percentage bar for tasks. Thanks to viewer Daniel BA1/4hler for this suggestion
import tflearn
import tensorflow as tf
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.estimator import regression
import matplotlib.pyplot as plt
TRAIN_DIR = '/home/student/Desktop/algonlty/trainmquokaggle'
TEST_DIR = '/home/student/Desktop/algonlty/testmquo'
#IMG_SIZE = 829
LR = 1e-3

MODEL_NAME = 'signxunsign-{}-{}.model'.format(LR, '2conv-basic') # just so we remember which saved model is which,

def label_img(img):
    word_label = img.split('.')[0]
    # conversion to one-hot array [sign,unsign]
   
    if word_label == 'sign': return [1,0]
   
    elif word_label == 'usign': return [0,1]

#this above function converts the data for us into array data of the image and its label.
def create_train_data():
    training_data = []
    for img in tqdm(os.listdir(TRAIN_DIR)):
        label = label_img(img)
        path = os.path.join(TRAIN_DIR,img)
        img = cv2.imread(path,cv2.IMREAD_GRAYSCALE)
        #img = cv2.resize(img, (829,67))                  #resize in the pixels of 829 x 67
        training_data.append([np.array(img),np.array(label)])
    shuffle(training_data)
    np.save('train_data.npy', training_data)
    return training_data

#processing of test data
def process_test_data():
    testing_data = []
    for img in tqdm(os.listdir(TEST_DIR)):
        path = os.path.join(TEST_DIR, img)
        img_num = img.split('.')[0]
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        #img = cv2.resize(img, (829, 67))
        testing_data.append([np.array(img), img_num])

    shuffle(testing_data)
    np.save('test_data.npy', testing_data)
    return testing_data

train_data = create_train_data()
test_data=process_test_data()
#now we are defining neural network to fit

tf.reset_default_graph()
convnet = input_data(shape=[None, 67, 829, 1], name='input')

convnet = conv_2d(convnet, 32, 5, activation='relu')
convnet = max_pool_2d(convnet, 5)

convnet = conv_2d(convnet, 64, 5, activation='relu')
convnet = max_pool_2d(convnet, 5)

convnet = conv_2d(convnet, 128, 5, activation='relu')
convnet = max_pool_2d(convnet, 5)

convnet = conv_2d(convnet, 64, 5, activation='relu')
convnet = max_pool_2d(convnet, 5)

convnet = conv_2d(convnet, 32, 5, activation='relu')
convnet = max_pool_2d(convnet, 5)

convnet = fully_connected(convnet, 1024, activation='relu')
convnet = dropout(convnet, 0.8)

convnet = fully_connected(convnet, 2, activation='softmax')
convnet = regression(convnet, optimizer='adam', learning_rate=LR, loss='categorical_crossentropy', name='targets')

model = tflearn.DNN(convnet, tensorboard_dir='log')

if os.path.exists('{}.meta'.format(MODEL_NAME)):
    model.load(MODEL_NAME)
    print('model loaded!')

train = train_data[:81] # 61x2, i[0]=67x829
test = train_data[:-30] # 51x2




#print(len(train[0]))
#print(train[1])

X = np.asarray([i[0] for i in train]) # 61x67x829
X=np.reshape(X,(81,67,829,1))
Y = np.asarray([i[1] for i in train]) # 61x2

test_x = np.asarray([i[0] for i in test])
print test_x.shape
test_x=np.reshape(test_x,(51,67,829,1))
test_y = np.asarray([i[1] for i in test])

model.fit({'input': X}, {'targets': Y}, n_epoch=5, validation_set=({'input': test_x}, {'targets': test_y}),snapshot_step=500, show_metric=True,batch_size=3, run_id=MODEL_NAME)

test_data = np.load('test_data.npy')
fig=plt.figure()

for num,data in enumerate(test_data[:42]):
    img_num = data[1]
    img_data = data[0]

    orig = img_data
    data = img_data.reshape(67,829) 
    data = np.reshape(data,(67,829,1))
    model_out = model.predict([data])

    model_out = model.predict([data])[0]
    if np.argmax(model_out) == 1:
        str_label = 'UnSigned'
        print(img_num+"==>"+str_label)
    else:
        str_label = 'Signed'
        print(img_num+"==>"+str_label)

