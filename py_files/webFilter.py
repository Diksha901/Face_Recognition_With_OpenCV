import cv2
import pandas as pd 
eye_cascade=cv2.CascadeClassifier('haarcascade_files//frontalEyes35x16.xml')
nose_cascade=cv2.CascadeClassifier('haarcascade_files//Nose18x15.xml')
img=cv2.imread('images//Before.jpg')
sunglasses=cv2.imread('images//glasses.png',-1)
moustache=cv2.imread('images//mustache.png',-1)

gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
eyes=eye_cascade.detectMultiScale(gray,1.3,5)
nose=nose_cascade.detectMultiScale(gray,1.3,5)

for (ex, ey, ew, eh) in eyes:
    spectacle_resized = cv2.resize(sunglasses, (int(0.99*ew), int(1.5*ew * sunglasses.shape[0] /sunglasses.shape[1])))
    for i in range(spectacle_resized.shape[0]):
        for j in range(spectacle_resized.shape[1]):
            if spectacle_resized[i, j][3] > 0:  # If not transparent
                img[ey + i, ex + j] = spectacle_resized[i, j][:3]

for (nx,ny,nw,nh) in nose:
    moustache_resized=cv2.resize(moustache,(nw,int(nh/2)))
    x_offset=int(1.1*nx)
    y_offset=int(1.1*(ny+nh//2))
    for i in range(moustache_resized.shape[0]):
        for j in range(moustache_resized.shape[1]):
            if moustache_resized[i,j][3]!=0:
                img[y_offset+i,x_offset+j]=moustache_resized[i,j][0:3]
    break            


img_flattened=img.reshape(-1,3)
dfa=pd.DataFrame(img_flattened,columns=["R","G","B"])
dfa.to_csv('results//Output.csv',index=False)
cv2.imshow('Modified Image', img)
cv2.waitKey(0)
cv2.destroyAllWindows()
 
cap=cv2.VideoCapture(0)
while True:
    ret,frame=cap.read()
    if not ret:
        break 
    gray_frame=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    eyes=eye_cascade.detectMultiScale(gray_frame,1.3,5)
    nose=nose_cascade.detectMultiScale(gray_frame,1.3,5)
    
    for (ex, ey, ew, eh) in eyes:
        spectacle_resized = cv2.resize(sunglasses, (int(ew), int(ew * sunglasses.shape[0] /sunglasses.shape[1])))
        for i in range(spectacle_resized.shape[0]):
            for j in range(spectacle_resized.shape[1]):
                if spectacle_resized[i, j][3] > 0:  # If not transparent
                    frame[ey + i, ex + j] = spectacle_resized[i, j][:3]
     
    for (nx,ny,nw,nh) in nose:
        moustache_resized=cv2.resize(moustache,(nw,int(nh/2)))
        x_offset=int(nx)
        y_offset=int((ny+nh//2))
        for i in range(moustache_resized.shape[0]):
            for j in range(moustache_resized.shape[1]):
                if moustache_resized[i,j][3]!=0:
                    frame[y_offset+i,x_offset+j]=moustache_resized[i,j][0:3]
        break   
    cv2.imshow('Live Filter',frame)
    if cv2.waitKey(1)&0xFF==ord('q'):
        break
    
cap.release()
cv2.destroyAllWindows()
    
    