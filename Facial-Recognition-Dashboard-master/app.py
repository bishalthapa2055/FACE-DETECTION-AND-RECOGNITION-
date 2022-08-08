from flask import Flask, redirect, jsonify, request, send_file, url_for, render_template
#from keras.backend.tensorflow_backend import set_session
#from utils.utils import *
#from utils.vgg_face import VGGFaceRecognizer
from PIL import Image, ImageDraw, ImageFont
from keras.backend.tensorflow_backend import set_session
import tensorflow as tf
from datetime import datetime
import numpy as np
import zipfile
import hashlib
import shutil
import sys
import os
from keras import applications 
from sklearn import metrics
from sklearn.metrics import confusion_matrix
from keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img 
from tensorflow.keras.models import Sequential 
from keras import optimizers
from keras.preprocessing import image
from keras.layers import Dropout, Flatten, Dense 
from keras import applications 
from keras.utils.np_utils import to_categorical
from werkzeug import secure_filename
import time
import cv2
from PIL import ExifTags
from tensorflow import keras
from keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img 
from keras.models import Sequential 
from keras import optimizers
from keras.preprocessing import image
from keras.layers import Dropout, Flatten, Dense 
import h5py
from tensorflow.keras import applications


# Configure Environment for GPU
#os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
#os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
#import tensorflow.compat.v1 as tf
#tf.disable_v2_behavior()
#sess = tf.Session()


#from tensorflow.python.keras import backend as K

#sess = tf.compat.v1.Session()
#K.set_session(sess)
#graph = tf.get_default_graph()
#set_session(sess)

at = None
content = None
uploaded_file = None
test_image = None
sfname = None
graph = None
PEOPLE_FOLDER = os.path.join('static/assets', 'recent-image')
# Some Hardcoded Values
model = keras.models.load_model('final-5-class.h5')

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

def classify(file_path):
    global label_packed
    global at
    global uploaded_file
    #imag = Image.open(file_path).convert("RGB")
    #image = face_cascade.detectMultiScale(image)
    #image=image.resize((200,200),Image.ANTIALIAS)
    #image=numpy.array(image.getdata()).reshape(-1,30,30,3)
    #image = image.img_to_array(image)
    #print(image.shape)
    #test_image=np.array(imag.getdata())



    img = Image.open(file_path)
    #img = np.asarray(img)
    #exif=dict((ExifTags.TAGS[k], v) for k, v in img._getexif().items() if k in ExifTags.TAGS)
    #if not exif['Orientation']:
     #   img=img.rotate(180, expand=True)
    try :
        for orientation in ExifTags.TAGS.keys() : 
            if ExifTags.TAGS[orientation]=='Orientation' : break 
        exif=dict(img._getexif().items())

        if   exif[orientation] == 3 : 
            img=img.rotate(180, expand=True)
        elif exif[orientation] == 6 : 
            img=img.rotate(270, expand=True)
        elif exif[orientation] == 8 : 
            img=img.rotate(90, expand=True)
    except:
        print("oh no")
    #from keras.applications.vgg16 import preprocess_input
    # prepare the image for the VGG model
    #imag = preprocess_input(img)
  
    imag = np.asarray(img)
    #a,b = imag.shape
    imag = cv2.resize(imag,(int(imag.shape[1]/2.5), int(imag.shape[0]/2.5)))

    imag = cv2.GaussianBlur(imag,(5,5),0)

        #imag = cv2.medianBlur(imag,5)
    img = np.asarray(imag)

        #show image from where face is extracted; 
    #cv2.imshow("immg",img)
    #cv2.waitKey()


    # directory = r'E:\S_E\Coading\project-iii\Facial-Recognition-Dashboard-master\Facial recognition\Facial-Recognition-Dashboard-master\static\assets\recent-image\test.jpg'
        #cv2.imwrite(directory,img)
        
        #imag=cv2.imread(file_path)

        #imag = cv2.cvtColor(imag,cv2.COLOR_RGB2GRAY) #gray scale conversion
        #imag=cv2.resize(imag,(500,500))
    faces=face_cascade.detectMultiScale(img,minSize =(80, 80))
    if faces==():
        at = "Difficult to detect face"
        exit()
    for (x, y, w, h) in faces:
        i = cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)
        roi_color = i[y:y+h, x:x+w]
            
        #test_image = image.load_img(roi_color, target_size = (200,200))
        #if image not found ?
        #print(len(roi_color))
    test_image=cv2.resize(roi_color,(200,200))
        #test_image=cv2.resize(roi_color,(200,200))
        #test_image = np.asarray(test_image)

        #show extracted face
    #cv2.imshow("img",test_image)
    #cv2.waitKey()
    #vgg16 = applications.VGG16(include_top=False, weights='imagenet')
    #datagen = ImageDataGenerator(rescale=1. / 255)

    #imag = img_to_array(test_image) 
    #imag = np.expand_dims(imag, axis=0)
    #imag /= 255.
    
    #peoples = ['bishal','ramesh','mitra', 'saugat','unknown']
    #time.sleep(.5)
    
    #bt_prediction = vgg16.predict(imag) 
    #preds = model.predict_proba(bt_prediction)
    #print(preds)
        
    #time.sleep(.5)
    #fin = max(max(preds))

    #if(fin == max(preds)[0]):
     #   at = 'Bishal : ' + str(100*(max(preds)[0]))
    #elif(fin == max(preds)[1]):
     #   at = 'Ramesh : ' + str(100*(max(preds)[1]))
    #elif(fin == max(preds)[2]):
     #   at = 'Mitra : ' + str(100*(max(preds)[2]))
    #elif(fin == max(preds)[3]):
     #   at = 'sauagt : ' + str(100*(max(preds)[3]))
    #elif(fin == max(preds)[4]):
     #   at = 'Unknown : ' + str(100*(max(preds)[4]))
 
    #else:
     #   at = "not done"
        #print(luna)
        #pred = model.predict_classes([image])[0]
        #result=model.predict(image)
        #sign = classes[pred+1]
        
    #classifier = keras.models.load_model("unknownfacepredict.h5")
    #test_image_luna = image.load_img(r'C:\Users\dell\Desktop\saugat\dummy\zebrish\237237_faces.jpg', target_size=(200,200))
    #test_image2 = image.img_to_array(test_image_luna)/255.
    #test_image2 = np.expand_dims(test_image2, axis=0)
    #luna = classifier.predict_proba(test_image2)
    test_image = image.img_to_array(test_image)/255.
    test_image = np.expand_dims(test_image, axis=0)
    luna = model.predict_proba(test_image)
    print(luna)
    #at = np.round(luna*100,2)
    #float(luna)[0]
    np.set_printoptions(suppress=True)
        #print(max(max(luna)))
    if max(max(luna)) == max(luna)[2]:
        if max(max(luna))>0.95:
           at = "Mitra Poudel"
        else:
            at="probably Mitra but not sure"
            
    elif max(max(luna)) == max(luna)[3]:
        if max(max(luna))>0.95:
            at = "Saugat Thapaliya"
        else:
            at="probably Saugat but not sure"

    elif max(max(luna)) == max(luna)[1]:
        if max(max(luna))>0.95:
            at = "Ramesh Pashwan"
        else:
            at="probably Ramesh but not sure"

    elif max(max(luna)) == max(luna)[0]:
        if max(max(luna))>0.95:
            at = "Bishal"
        else:
            at="probably Bishal but not sure"
        
            
    elif max(max(luna)) == max(luna)[4]:
        if max(max(luna))>0.95:
            at = "Unknown"
        else:
            at="Unknown"

    else:
        at="super"


    
    
    #except:
        #at="size of image is very small"     
    



#some hardcode
ROOT_DATA_DIR = 'static/data'
CROPPED_FACES_DIR = os.path.join(ROOT_DATA_DIR, 'cropped_faces')
RECOGNIZED_FACES_DIR = os.path.join(ROOT_DATA_DIR, 'recognized_faces')
#mtcnn = MTCNN()
#face_recognizer = VGGFaceRecognizer(model='senet50')

# Flask Config
app = Flask(__name__)
app.config['DEBUG'] = True
#app.config["TEMPLATES_AUTO_RELOAD"] = True

# In App Variables
content = list()


@app.route('/', methods=['GET', 'POST'])
@app.route('/home', methods=['GET', 'POST'])
def home():
    if request.method == 'GET':
        return render_template('home.html')

    return redirect(url_for('upload_faces'))


@app.route('/about', methods=['GET'])
def about():
    return render_template('about.html', title="About")


@app.route('/handleerror', methods=['GET'])
def handleerror():
    return render_template('handleerror.html', title="About")

@app.route('/trainnewpeople', methods=['GET','POST'])
def trainnewpeople():
    #if request.method == 'GET':
     #   return render_template('trainnewpeople.html')

    return redirect(url_for('upload_faces2'))
    





@app.route('/contact', methods=['GET'])
def contact():
    return render_template('contact.html', title="Contact")


@app.route('/upload_faces', methods=['POST', 'GET'])
def upload_faces():
    global content, graph
    global uploaded_file,sfname
    # Display Page
    if request.method == 'GET':
        return render_template('upload_faces.html')

    # Delete Existing Images
    try:
        shutil.rmtree(CROPPED_FACES_DIR)
    except:
        pass

    try:
        imagee = request.files['file']
   
    #pil_image = Image.open(uploaded_file)
    #uploaded_file =  Image.open(uploaded_file)
    #print("yyyyyyyyyyyyy" + str(pil_image))
    
        sfname = 'static/assets/img/'+str(secure_filename("test.jpg"))
        imagee.save(sfname)
        at = classify(sfname)
    
    #result = classify(uploaded_file)
    #print(result)

   # sfname = './static/assets/recent-image/'+str(secure_filename("test.jpg"))
    #uploaded_file.save(sfname)

    # Making sure Folder exists
    #if not os.path.isdir(CROPPED_FACES_DIR):
     #   os.makedirs(CROPPED_FACES_DIR)

    # Get uploaded files
    #uploaded_files = request.files.getlist("file")

    #face_id = 0

    # Detect Faces for each Uploaded Image
    #content = list()
    #for uploaded_file in uploaded_files:
     #   image = Image.open(uploaded_file).convert('RGB')
      #  image = np.asarray(image)

        # Detect Faces using MTCNN
       # with graph.as_default():
        #    set_session(sess)
         #   detected_faces = mtcnn.detect_faces(image)
          #  print(detected_faces)

        #height, width, _ = image.shape
        #for detected_face in detected_faces:
         #   x1, y1, x2, y2 = fix_coordinates(
          #      detected_face['box'], width, height
           # )
            #cropped_face = image[y1:y2, x1:x2]
            #try:
             #   cropped_face = Image.fromarray(cropped_face)
            #except:
             #   continue
            #saved_image_path = os.path.join(
             #   CROPPED_FACES_DIR,
              #  rem_punctuation(f"{face_id}_{uploaded_file.filename}")
            #)
            #cropped_face.save(saved_image_path)
            #content.append(
             #   {
              #      'id': face_id,
               #     'image': f"/{saved_image_path}"
                #}
            #)
            #face_id += 1

        return redirect(url_for('checkfinalresult'))
    except:
        return redirect(url_for('handleerror'))

app.config['UPLOAD_FOLDER'] = PEOPLE_FOLDER
@app.route('/checkfinalresult')
def checkfinalresult():
    #full_filename = os.path.join(app.config['UPLOAD_FOLDER'], )
    #name = at
    #imgpath = r'E:\S_E\Coading\project-iii\Facial-Recognition-Dashboard-master\Facial recognition\Facial-Recognition-Dashboard-master\static\assets\recent-image\test.jpg'
    #imaage = cv2.imread(imgpath)
    #full_filename = os.path.join(app.config['UPLOAD_FOLDER'], 'test.jpg')
    #test_imagee = Image.open(test_image)
    #uploaded_file = Image.open(uploaded_file)
    #print(img)
    #img = Image.open(uploaded_file)
    #img = np.asarray(uploaded_file)
    #return render_template('finalresult.html', user_image = full_filename, user_name=at)
    return render_template('checkfinalresult.html', user_name=at,sfname = sfname, image_path=sfname)

@app.route('/upload_faces2', methods=['POST', 'GET'])
def upload_faces2():
    global content
    # Display Page
    if request.method == 'GET':
        return render_template('upload_faces2.html')
    #uploaded_file2 = request.files['file']
    # Delete Existing Images
    try:
        shutil.rmtree(CROPPED_FACES_DIR)
    except:
        pass

    # Making sure Folder exists
    if not os.path.isdir(CROPPED_FACES_DIR):
        os.makedirs(CROPPED_FACES_DIR)

    # Get uploaded files
    uploaded_files = request.files.getlist("file")
    #print(uploaded_files)
    face_id = 0

    # Detect Faces for each Uploaded Image
    content = list()
    for uploaded_file in uploaded_files:
        image = Image.open(uploaded_file).convert('RGB')
        image = np.asarray(image)
        # Detect Faces using MTCNN
        face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
        faces = face_cascade.detectMultiScale(image)

        height, width, _ = image.shape
        for (x, y, w, h) in faces:
            img = cv2.rectangle(image, (x, y), (x+w, y+h), (255, 0, 0), 2)
            roi_color = img[y:y+h, x:x+w]
            
        #for detected_face in detected_faces:
         #   x1, y1, x2, y2 = fix_coordinates(
          #      detected_face['box'], width, height
           # )
            #cropped_face = image[y1:y2, x1:x2]
            try:
                roi_color = Image.fromarray(roi_color)
            except:
                continue
            saved_image_path = os.path.join(
                CROPPED_FACES_DIR,
                (f"{face_id}_{uploaded_file.filename}")
            )
            roi_color.save(saved_image_path)
            content.append(
                {
                    'id': face_id,
                    'image': f"/{saved_image_path}"
                }
            )
            face_id += 1

    return redirect(url_for('label_faces'))








@app.before_request
def before_request():
    # When you import jinja2 macros, they get cached which is annoying for local
    # development, so wipe the cache every request.
    if 'localhost' in request.host_url or '0.0.0.0' in request.host_url:
        app.jinja_env.cache = {}

@app.route('/label_faces', methods=['POST', 'GET'])
def label_faces():
    global content, graph

    # Display Page
    if request.method == 'GET':
        return render_template('label_faces.html', title="Label", table_contents=content)

    face_list = list()
    label_list = list()
    for element in content:
        face_id = element.get('id')
        image_path = element.get('image')
        label = request.form.get(f"face-name-{face_id}")
        print(face_id, image_path, label)

        if not label:
            continue

        # Create the List to pass to FR Module
        image = Image.open(image_path[1:]).convert('RGB')
        face_list.append(image)
        label_list.append(label)

    # Register Faces - Extract and store ground truth Facial Features
    #with graph.as_default():
     #   set_session(sess)
      #  face_recognizer.register_faces(face_list, label_list)

    #return redirect(url_for('upload_pictures'))
    return redirect(url_for('upload'))

@app.route('/upload_pictures', methods=['POST', 'GET'])
def upload_pictures():
    global graph

    # Display Page
    if request.method == 'GET':
        return render_template('upload_pictures.html', title="Upload")

    # Delete Existing Images
    try:
        shutil.rmtree(RECOGNIZED_FACES_DIR)
    except:
        pass

    # Making sure Folder exists
    if not os.path.isdir(RECOGNIZED_FACES_DIR):
        os.makedirs(RECOGNIZED_FACES_DIR)

    # Get uploaded files
    uploaded_files = request.files.getlist("file")

    # Detect Faces for each Uploaded Image
    recognized_image_list = list()
    for uploaded_file in uploaded_files:
        image_PIL = Image.open(uploaded_file).convert('RGB')
        image = np.asarray(image_PIL)

        # Detect Faces using MTCNN
        with graph.as_default():
            set_session(sess)
            detected_faces = mtcnn.detect_faces(image)

        height, width, _ = image.shape
        font_size = int(np.mean([width, height]) * 0.03)

        # TODO: From here down
        draw = ImageDraw.Draw(image_PIL)
        for detected_face in detected_faces:
            x1, y1, x2, y2 = fix_coordinates(
                detected_face['box'], width, height
            )
            cropped_face = image[y1:y2, x1:x2]
            try:
                cropped_face = Image.fromarray(cropped_face)
            except:
                continue

            with graph.as_default():
                set_session(sess)
                face_name = face_recognizer.recognize(
                    cropped_face, thresh=0.30)
            if face_name:
                draw.rectangle((x1, y1, x2, y2), width=5)
                font = ImageFont.truetype(
                    "static/assets/fonts/Roboto-Regular.ttf", font_size)

                text_w, text_h = draw.textsize(face_name)
                text_x = int(x1)
                text_y = int(y1 - font_size)
                draw.text((text_x, text_y), face_name, fill='red', font=font)

        image_path = os.path.join(
            RECOGNIZED_FACES_DIR, rem_punctuation(
                f"{time.time()}_{uploaded_file.filename}")
        )
        image_PIL.save(image_path)
        recognized_image_list.append('/'+image_path)

    return redirect(url_for('display_results'))


@app.route('/display_results', methods=['GET'])
def display_results():
    # Get List of images in Recognized Faces Directory
    images = os.listdir(RECOGNIZED_FACES_DIR)
    image_paths = ['/'+os.path.join(RECOGNIZED_FACES_DIR, im_path).replace(' ', '%20')
                   for im_path in images]

    return render_template('display_results.html', image_paths=image_paths)


if __name__ == '__main__':
    app.jinja_env.auto_reload = True
    app.config['TEMPLATES_AUTO_RELOAD'] = True
    app.run(debug=True,use_reloader=True,threaded = False, host='0.0.0.0',port=5000)

#    from livereload import Server
 #   server = Server(app.wsgi_app)
  #  server.serve(host = '0.0.0.0',port=5000)


#    app.run(
 #       host='0.0.0.0',
  #      port='5000',
   #     use_reloader=False,
    #    threaded = True
    #)
