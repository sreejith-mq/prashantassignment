-->Here given assignment problem is  classification problem to sort documents into two categorries - Signed and Unsigned.
-->I am using CNN based supervised training approach.
-->To feed given signed and unsigned training images to CNN, Images should have same dimensions else we need to pad with default pixel values.
-->Here I am trasforming each image to fixed size 829 x 67 which is a largest dimension in training.
-->2 files are there one for training and other for resizing images.
-->After training out put will be like "image number"==>"signed/unsigned"
-->eg. 15==>signed
-->order of images will be shuffled because I have shuffled testdata in testing.
-->for preparing test data set I have taken 21 images from each signed and unsigned dataset.
-->Here I have done slight modification , rather than passing one image during prediction. I am making prediction on all 42 test          images as I have mentioned that 21 signed and 21 unsigned in testing. 
-->test data set is  available on below mentioned link
https://drive.google.com/open?id=1wfJLLacYgcbZYPK-JavcBKBsg34BI03S
-->training data is availbsle on below mentioned link
https://drive.google.com/open?id=1_QG4-SyGbsUFe2QCO1SGfpOJ_nZOqDVs
