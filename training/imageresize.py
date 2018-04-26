
import cv2   
from tqdm import tqdm 
import os 
TRAIN_DIR = '/home/student/Desktop/mquotient/trainmquokaggle'
TEST_DIR = '/home/student/Desktop/mquotient/testmquo'

	

for img in tqdm(os.listdir(TRAIN_DIR)):
		label = label_img(img)
		path = os.path.join(TRAIN_DIR,img)
		img = cv2.imread(path,cv2.IMREAD_GRAYSCALE)
		res = cv2.resize(img,(829, 67), interpolation = cv2.INTER_CUBIC)
		cv2.imwrite( "../../images/"+label+".jpg", res );

for img in tqdm(os.listdir(TEST_DIR)):
		label = label_img(img)
		path = os.path.join(TEST_DIR,img)
		img = cv2.imread(path,cv2.IMREAD_GRAYSCALE)
		res = cv2.resize(img,(829, 67), interpolation = cv2.INTER_CUBIC)
		cv2.imwrite( "../../images/"+label+".jpg", res );
