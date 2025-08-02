#  Face Recognition with OpenCV  

* The project aims on developing a Face Recognition and Face Data Collection system .
* The project can be used for face lock authentication on various smart electronic gadgets .  
* The project also develops a WebFilter.py to place a mustache and sunglass on face of the
  user in realtime like the filters we use in various photo sharing related applications .   

---

##  Files in This Repository

- *`haarcascade_files`* — The directory containing all the files to detect various features of the face like eyes, nose etc. 
- *`py_files`* - Directory containing files in which the Face Recognition code is written **face_recognition.py** ,**face_data_collection.py**,**webFilter.py**
- *`images`* - Directory containing files in which we have images of sunglasses and mustache to be overlayed on the user's face and also image of a person to test the application of the filter over it . 
- `requirements.txt` - The list of dependencies(python,sqlalchemy,psycopg2 ...etc.)  requirerd to run the application 
- `README.md` — You’re reading it!

# Project Structure :
face_recognition/

  
--->`haarcascade_files/`

      ------>frontalEyes35x16.xml 
      
      ------>haarcascade_frontalface_alt.xml
      
      ------>haarcascade_frontalface_default.xml
      
      ------>haarcascade_mcs_eyepair_big.xml
      
      ------>haarcascade_mcs_nose.xml
      
      ------>Nose18x15.xml

------>`images/`

       ---->before.png
       
       ---->glasses.png
       
       ---->mustache.png
       
       
------>`py_files/`

       ---->face_data_collection.py
       
       ---->face_recognition.py
       
       ---->webFilter.py
       
---->venv(gitignored) 

---->requirements.txt

---->README.md


---




##  TechStack 



> `Python`  :     Main programming language for scripting face recognition, data collection, and filter application

> `OpenCV (cv2)`:   Image processing, face/eye/nose detection using Haar cascades, drawing, filter overlays

> `NumPy`    :       Efficient matrix operations to flatten images and work with images as numpy arrays

> `Haar Cascades`  : Pre-trained classifiers for detecting facial features (OpenCV `.xml` files)                       



  

---
