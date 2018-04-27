import cv2                
import numpy as np        
import os                 
from random import shuffle 
from tqdm import tqdm      
import tflearn
import tensorflow as tf
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.estimator import regression
import matplotlib.pyplot as plt
TRAIN_DIR = '/home/student/Desktop/algonlty/trainmquo'   #training directory
TEST_DIR = '/home/student/Desktop/algonlty/testmquo'      #testing directory  

LR = 1e-3

MODEL_NAME = 'signxunsign-{}-{}.model'.format(LR, '2conv-basic') 

def label_img(img):
    word_label = img.split('.')[0]
    # conversion to one-hot array [sign,unsign]
    #                            [1 for sign]
    if word_label == 'sign': return [1,0]
    #                             [0 for unsign]
    elif word_label == 'unsign': return [0,1]

#preparing training data
def create_train_data():
    training_data = []
    for img in tqdm(os.listdir(TRAIN_DIR)):
        label = label_img(img)
        path = os.path.join(TRAIN_DIR,img)
        img = cv2.imread(path,cv2.IMREAD_GRAYSCALE)
           
        training_data.append([np.array(img),np.array(label)]) # preparing training data as 2d list where 1st numpy array of image and 2nd is label in 0/1
    shuffle(training_data) # shuffle training for leveraging more randomness
    np.save('train_data.npy', training_data)
    return training_data

#preparing test data
def process_test_data():
    testing_data = []
    for img in tqdm(os.listdir(TEST_DIR)):
        path = os.path.join(TEST_DIR, img)
        img_num = img.split('.')[0]
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        
        testing_data.append([np.array(img), img_num]) # preparing training data as 2d list where 1st numpy array of image and 2nd is image number

    shuffle(testing_data) # shuffle testing for levaraging more randomness
    np.save('test_data.npy', testing_data)
    return testing_data

train_data = create_train_data()
test_data=process_test_data()


#defining CNN
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

if os.path.exists('/home/student/Desktop/mquotient/mquo model{}.meta{}.meta'.format(MODEL_NAME)):
    model.load(MODEL_NAME)
    print('model loaded!')
 
#preparing trainig aand validation sets
    
train = train_data[:81] # 61x2, i[0]=67x829
test = train_data[:-30] # 51x2





#loading training and validations sets
X = np.asarray([i[0] for i in train]) # 61x67x829
X=np.reshape(X,(81,67,829,1))
Y = np.asarray([i[1] for i in train]) # 61x2

test_x = np.asarray([i[0] for i in test])

test_x=np.reshape(test_x,(51,67,829,1))  # preparing validation images
test_y = np.asarray([i[1] for i in test]) # preparing validation labels

#training model
model.fit({'input': X}, {'targets': Y}, n_epoch=20, validation_set=({'input': test_x}, {'targets': test_y}),snapshot_step=500, show_metric=True,batch_size=3, run_id=MODEL_NAME)

