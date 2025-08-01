import cv2
import pandas as pd

image_path = 'images//Before.png'  
image = cv2.imread(image_path)

face_cascade = cv2.CascadeClassifier('haarcascade_files//haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('haarcascade_files//haarcascade_mcs_eyepair_big.xml')
nose_cascade = cv2.CascadeClassifier('haarcascade_files//haarcascade_mcs_nose.xml')

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

faces = face_cascade.detectMultiScale(gray, 1.3, 5)


sunglasses = cv2.imread('images//glasses.png', -1)
mustache = cv2.imread('images//mustache.png', -1)

for (x, y, w, h) in faces:
    roi_gray = gray[y:y + h, x:x + w]
    roi_color = image[y:y + h, x:x + w]

    eyes = eye_cascade.detectMultiScale(roi_gray)
    for (ex, ey, ew, eh) in eyes[:2]:  
        sunglasses_resized = cv2.resize(sunglasses, (ew, eh))
        for i in range(sunglasses_resized.shape[0]):
            for j in range(sunglasses_resized.shape[1]):
                if sunglasses_resized[i, j, 3] != 0:  
                    roi_color[ey + i, ex + j] = sunglasses_resized[i, j, :3]

    noses = nose_cascade.detectMultiScale(roi_gray)
    for (nx, ny, nw, nh) in noses[:1]:  
        mustache_resized = cv2.resize(mustache, (nw, int(nh / 2)))
        for i in range(mustache_resized.shape[0]):
            for j in range(mustache_resized.shape[1]):
                if mustache_resized[i, j, 3] != 0:  
                    roi_color[ny + int(nh / 2) + i, nx + j] = mustache_resized[i, j, :3]


flattened_image = image.reshape(-1, 3)
df = pd.DataFrame(flattened_image, columns=['R', 'G', 'B'])
df.to_csv('results//output_image.csv', index=False)


cv2.imshow('Modified Image', image)
cv2.waitKey(0)
cv2.destroyAllWindows()
