import numpy as np
import cv2 
import os
def distance(v1,v2):
  return np.sqrt((v1-v2)**2).sum()
def knn(train,test,k=5):
  dist=[]
  for i in range(train.shape[0]):
    ix=train[i,:-1]
    iy=train[i,-1]
    d=distance(test,ix)
    dist.append([d,iy])

  dk=sorted(dist,key=lambda x:x[0])[:k]
  labels=np.array(dk)[:,-1]
  output=np.unique(labels,return_counts=True)
  index=np.argmax(output[1])
  return output[0][index]

cap=cv2.VideoCapture(0)
face_cascade=cv2.CascadeClassifier("haarcascade_files//haarcascade_frontalface_alt.xml")
eye_cascade=cv2.CascadeClassifier("haarcascade_files//frontalEyes35x16.xml")
nose_cascade=cv2.CascadeClassifier("haarcascade_files//Nose18x15.xml")
skip=0
dataset_path='./data/'
face_data=[]
labels=[]
class_id=0
names={}
for fx in os.listdir(dataset_path):
  if fx.endswith('.npy'):
    names[class_id]=fx[:-4]
    data_item=np.load(dataset_path+fx)
    face_data.append(data_item)
    target=class_id*np.ones((data_item.shape[0],))
    class_id+=1
    labels.append(target)

print(len(face_data))
face_dataset=np.concatenate(face_data,axis=0)
face_labels=np.concatenate(labels,axis=0).reshape((-1,1))
print(face_dataset.shape)
print(face_labels.shape)
trainset=np.concatenate((face_dataset,face_labels),axis=1)
print(trainset.shape)

##Testing data
while True:
  ret,frame=cap.read()
  if ret==False:
    continue

  faces=face_cascade.detectMultiScale(frame,1.3,5)
  eyes = eye_cascade.detectMultiScale(frame, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
  nose = nose_cascade.detectMultiScale(frame, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
  if(len(faces)==0):
    continue

  for face in faces[-1:]:
    x,y,w,h=face

    offset=10
    face_section=frame[y-offset:y+h+offset,x-offset:x+w+offset]
    face_section=cv2.resize(face_section,(100,100))
    out=knn(trainset,face_section.flatten())
    pred_name=names[int(out)]
    cv2.putText(frame, pred_name, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
    cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,255),2)
  for (x, y, w, h) in eyes:
    cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)

  # Draw rectangles around detected nose
  for (x, y, w, h) in nose:
      cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
  cv2.imshow("faces",frame)
  key_pressed=cv2.waitKey(1) &0xFF
  if key_pressed==ord('q'):
    break

cap.release()
cv2.destroyAllWindows()