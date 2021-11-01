import cv2,os
import numpy
as np
from keras.utils import np_utils
data_path='C:\\Users\\MYPC\\Projects\\face-mask-detector\\dataset'
categories=os.listdir(data_path)
labels=[i for i in range(len(categories))]
label_dict=dict(zip(categories,labels))
print(label_dict)
print(categ
ories)
print(labels
) data=[]
target=[]
for category in categories: folder_path=os.path.join(data_path,category)
img_names=os.listdir(folder_path)
for img_name in img_names:
mg_path=os.path.join(folder_path,img_name)
img=cv2.imread(img_path)
try:
gray=cv2.cvtColor(img,cv2.COLOR_BGR2G
RAY)
#Coverting the image into gray scale
resized=cv2.resize(gray,(100,100))
#resizing the gray scale into 100x100, since we need a fixed common
size for all the images in the dataset
data.append(resized)
target.append(label_dict[category])
#appending the image and the label(categorized) into the list
(dataset) except Exception as e:
print('Exception:',e)
mymodel=load_model('leo.h5')
cap=cv2.VideoCapture(0)
face_cascade=cv2.CascadeClassifier('haarcascade_frontalface_default.
xml')
while cap.isOpened():
_,img=cap.read()
face=face_cascade.detectMultiScale(img,scaleFactor=1.1,minNeigh
bors=4)
for(x,y,w,h) in face:
face_img = img[y:y+h, x:x+w] cv2.imwrite('temp.jpg',face_img)
test_image=image.load_img('temp.jpg',target_size=(150,150,3))
test_image=image.img_to_array(test_image)
test_image=np.expand_dims(test_image,axis=0)
pred=mymodel.predict_classes(test_image)[0][0]
if pred==1:
cv2.rectangle(img,(x,y),(x+w,y+h),(0,0,255),3)
cv2.putText(img,'NOMASK',((x+w)//2,y+h+20),cv2.FONT_HE
RSHEY_SIMPLEX,1,(0,0,255),3)
else:
cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),3)
cv2.putText(img,'MASK',((x+w)//2,y+h+20),cv2.FONT_HERSHEY_SI
MPLE X,1,(0,255,0),3)
datet=str(datetime.datetime.now())
cv2.putText(img,datet,(400,450),cv2.FONT_HERSHEY_SIMPLEX,0.5,
(255,2 55,255),1)
cv2.imshow('img',img)
If
cv2.waitKey(1)==ord('q'):
break
cap.release()
cv2.destroyAllWindows()
