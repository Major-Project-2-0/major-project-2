"""
This is the main python code for smart attendance system 
which consists of flash web app and Face Recognition System using KNN classifier

"""
import os
from datetime import date
from datetime import datetime
from flask import Flask,request,render_template
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
import pandas as pd
import cv2
import joblib

#### Defining Flask App
app = Flask(__name__)


#### Saving Date today in 2 different formats
datetoday = date.today().strftime("%m_%d_%y")
datetoday2 = date.today().strftime("%d-%B-%Y")


#### Initializing VideoCapture object to access WebCam
face_detector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

try:
    cap = cv2.VideoCapture(1)
except Exception:
    cap = cv2.VideoCapture(0)


#### If these directories don't exist, create them
if not os.path.isdir('Attendance'):
    os.makedirs('Attendance')
if not os.path.isdir('static'):
    os.makedirs('static')
if not os.path.isdir('static/faces'):
    os.makedirs('static/faces')
if f'Attendance-{datetoday}.csv' not in os.listdir('Attendance'):
    with open(f'Attendance/Attendance-{datetoday}.csv','w',encoding="UTF-8") as f:
        f.write('Student-count,Name,Enrollment,Time,Subject-Name,Class-Type,Faculty-Name,Class-Scheduled-Time')



# get a number of total registered users

def totalreg():
    return len(os.listdir('static/faces'))


#### extract the face from an image
def extract_faces(img):
    if img!=[]:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        face_points = face_detector.detectMultiScale(gray, 1.3, 5)
        return face_points
    else:
        return []

#### Identify face using ML model
def identify_face(facearray):
    model = joblib.load('static/face_recognition_model.pkl')
    return model.predict(facearray)


#### A function which trains the model on all the faces available in faces folder
def train_model():
    faces = []
    labels = []
    userlist = os.listdir('static/faces')
    for user in userlist:
        for imgname in os.listdir(f'static/faces/{user}'):
            img = cv2.imread(f'static/faces/{user}/{imgname}')
            resized_face = cv2.resize(img, (50, 50))
            faces.append(resized_face.ravel())
            labels.append(user)
    faces = np.array(faces)
    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(faces,labels)
    joblib.dump(knn,'static/face_recognition_model.pkl')


#### Extract info from today's attendance file in attendance folder
def extract_attendance():
    df = pd.read_csv(f'Attendance/Attendance-{datetoday}.csv')
    studentcount = df['Student-count']
    names = df['Name']
    rolls = df['Enrollment']
    times = df['Time']
    subname = df['Subject-Name']
    classtype = df['Class-Type']
    facultyname = df['Faculty-Name']
    schdtime = df['Class-Scheduled-Time']
    l = len(df)
    return studentcount,names,rolls,times,subname,classtype,facultyname,schdtime,l


#### Add Attendance of a specific user
def add_attendance(name, studentcount, subname, classtype, facultyname, schdtime):
    username = name.split('_')[0]
    userid = name.split('_')[1]
    current_time = datetime.now().strftime("%H:%M:%S")
    usercount = studentcount
    usersubname = subname
    userclasstype = classtype
    userfaculty = facultyname
    userschdtime = schdtime
    
    df = pd.read_csv(f'Attendance/Attendance-{datetoday}.csv')
    if int(userid) not in list(df['Enrollment']):
        with open(f'Attendance/Attendance-{datetoday}.csv','a', encoding="UTF-8") as f:
            f.write(f'\n{usercount},{username},{userid},{current_time},{usersubname},{userclasstype},{userfaculty},{userschdtime}')


################## ROUTING FUNCTIONS #########################

#### Our main page
@app.route('/')
def home():
    studentcount,names,rolls,times,subname,classtype,facultyname,schdtime,l = extract_attendance()    
    return render_template('home.html',studentcount=studentcount,names=names,rolls=rolls,times=times,subname=subname,classtype=classtype,facultyname=facultyname,schdtime=schdtime,l=l,totalreg=totalreg(),datetoday2=datetoday2) 


@app.route('/start',methods=['GET','POST'])
def start():
    if 'face_recognition_model.pkl' not in os.listdir('static'):
        return render_template('home.html',totalreg=totalreg(),datetoday2=datetoday2,mess='There is no trained model in the static folder. Please add a new face to continue.')

    subname = request.form.get('subname')
    classtype = request.form.get('classtype')
    facultyname = request.form.get('facultyname')
    schdtime = request.form.get('schdtime')

    i=1
    cap = cv2.VideoCapture(0)
    ret = True
    while ret:
        ret,frame = cap.read()
        if extract_faces(frame)!=():
            (x,y,w,h) = extract_faces(frame)[0]
            cv2.rectangle(frame,(x, y), (x+w, y+h), (255, 0, 20), 2)
            face = cv2.resize(frame[y:y+h,x:x+w], (50, 50))
            identified_person = identify_face(face.reshape(1,-1))[0]
            add_attendance(name=identified_person,studentcount=i,subname=subname,classtype=classtype,facultyname=facultyname,schdtime=schdtime)
            i=i+1
            cv2.putText(frame,f'{identified_person}',(30,30),cv2.FONT_HERSHEY_SIMPLEX,1,(255, 0, 20),2,cv2.LINE_AA)
        cv2.imshow('Attendance',frame)
        if cv2.waitKey(1)==27:
            break
    cap.release()
    cv2.destroyAllWindows()
    studentcount,names,rolls,times,subname,classtype,facultyname,schdtime,l = extract_attendance()    
    return render_template('home.html',studentcount=studentcount,names=names,rolls=rolls,times=times,subname=subname,classtype=classtype,facultyname=facultyname,schdtime=schdtime,l=l,totalreg=totalreg(),datetoday2=datetoday2) 

#### This function will run when we add a new user
@app.route('/add',methods=['GET','POST'])
def add():
    newusername = request.form.get['newusername']
    newuserid = request.form.get['newuserid']
    userimagefolder = 'static/faces/'+newusername+'_'+str(newuserid)
    if not os.path.isdir(userimagefolder):
        os.makedirs(userimagefolder)
    cap = cv2.VideoCapture(0)
    i,j = 0,0
    while 1:
        _,frame = cap.read()
        faces = extract_faces(frame)
        for (x,y,w,h) in faces:
            cv2.rectangle(frame,(x, y), (x+w, y+h), (255, 0, 20), 2)
            cv2.putText(frame,f'Images Captured: {i}/50',(30,30),cv2.FONT_HERSHEY_SIMPLEX,1,(255, 0, 20),2,cv2.LINE_AA)
            if j%10==0:
                name = newusername+'_'+str(i)+'.jpg'
                cv2.imwrite(userimagefolder+'/'+name,frame[y:y+h,x:x+w])
                i+=1
            j+=1
        if j==500:
            break
        cv2.imshow('Adding new User',frame)
        if cv2.waitKey(1)==27:
            break
    cap.release()
    cv2.destroyAllWindows()
    print('Training Model')
    train_model()
    studentcount,names,rolls,times,subname,classtype,facultyname,schdtime,l = extract_attendance()
    return render_template('home.html',studentcount=studentcount,names=names,rolls=rolls,times=times,subname=subname,classtype=classtype,facultyname=facultyname,schdtime=schdtime,l=l,totalreg=totalreg(),datetoday2=datetoday2)


#### Our main function which runs the Flask App
if __name__ == '__main__':
    app.run(debug=True)
