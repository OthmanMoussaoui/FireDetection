# -*- coding: utf-8 -*-
"""
Created on Sun Jan 15 14:34:34 2023

@author: otman
"""

# -*- coding: utf-8 -*-
"""
Created on Sun Jan 15 12:24:02 2023

@author: otman
"""

# Python program to open the
# camera in Tkinter
# Import the libraries,
# tkinter, cv2, Image and ImageTk

from tkinter import *
import cv2
from PIL import Image,ImageDraw,ImageGrab,ImageTk
import numpy as np
import tensorflow as tf
from keras.preprocessing import image

model = tf.keras.models.load_model('output/fire_detection.h5')
# Define a video capture object

video = cv2.VideoCapture(0)


# Create a GUI app
app = Tk()
app.resizable(0,0)
app.title("Firedetection_ENSAB") 

# Bind the app with Escape keyboard to
# quit app whenever pressed
app.bind('<Escape>', lambda e: app.quit())

lbl = Label(app, text="Hello this is an app for Detecting Fire", font=("Tekton Pro", 10))
lbl.grid(column=0, row=1,sticky='nsew')
lbl.config(justify='center')

image=ImageTk.PhotoImage(Image.open("logo-ensa-berrechid.png"))

# Create a Label widget and set its image attribute to the PhotoImage object
label = Label(app, image=image)
label.grid(row=0, column=0,sticky='')
label.config(justify='center')

# Use the grid geometry manager to place the label in the top-left corner of the window


# Create a label and display it on app
label_widget = Label(app)
label_widget.grid(row=2)

# Create a function to open camera and
# display it in the label_widget on app


def open_camera():
	# Capture the video frame by frame
    _, frame = video.read()
#Convert the captured frame into RGB
    im = Image.fromarray(frame, 'RGB')
#Resizing into 224x224 because we trained the model with this image size.
    im = im.resize((128,128))
    img_array = image.img_to_array(im)
    img_array = np.expand_dims(img_array, axis=0) / 255
    probabilities = model.predict(img_array)[0]
        #Calling the predict method on model to predict 'fire' on the image
    prediction = np.argmax(probabilities)
        #if prediction is 0, which means there is fire in the frame.
    if prediction == 0:
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        print(probabilities[prediction])
        prob = int(probabilities[prediction]*100)
        cv2.putText(frame, f'Fire: {prob}%', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
    else:
        print("no fire")
        cv2.putText(frame, f'No Fire', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
   
    cv2.imshow("Detecting", frame)
    key=cv2.waitKey(1)
	# Capture the latest frame and transform to image
	# Repeat the same process after every 10 seconds
    label_widget.after(10, open_camera)

image1=ImageTk.PhotoImage(Image.open("fire.png"))


# Create a Label widget and set its image attribute to the PhotoImage object
label1 = Label(app, image=image1)

# Use the grid geometry manager to place the label in the top-left corner of the window
label1.grid(row=2, column=0,sticky='')
# Create a button to open the camera in GUI app
lbl = Label(app, text="by Othman Moussaoui", font=("Tekton Pro", 10))
lbl.grid(column=0, row=3,sticky='nsew')
lbl = Label(app, text="Supervised by Pr:Lahcen MOUMOUN", font=('Helvetica' , 10,'bold'))
lbl.grid(column=0, row=4,sticky='nsew')
button1 = Button(app, text="Start detect", command=open_camera)
button1.grid(row=5)

# Create an infinite loop for displaying app on screen
app.mainloop()
