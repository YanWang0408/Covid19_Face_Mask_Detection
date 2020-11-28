# import the necessary packages
import os
import cv2
import time
import imutils
import numpy as np
import tensorflow as tf


from keras.applications.mobilenet_v2 import preprocess_input
from keras.preprocessing.image import img_to_array
from keras.models import load_model
from imutils.video import VideoStream

import cv2
import ssl
import numpy as np
import pyautogui
import imutils
import smtplib

from email.mime.text import MIMEText
from email.utils import formataddr
from email.mime.multipart import MIMEMultipart  # New line
from email.mime.base import MIMEBase  # New line
from email import encoders  # New line




def detect_and_predict_mask(frame, faceNet, maskNet):
	# grab the dimensions of the frame and then construct a blob from it
	# a blob is just a of image with the same spatial dimensions (i.e., width and height), same depth (number of channels), that have all be preprocessed in the same manner.
	(h, w) = frame.shape[:2]
	# blob = cv2.dnn.blobFromImage(image, scalefactor=1.0, size, RGBmean, swapRB=True) 
	# The mean RGB values are the means for each individual RGB channel across all images in your training set
	blob = cv2.dnn.blobFromImage(frame, 1, (224, 224), (104, 177, 123)) 
	# blob.shape = (num_images=1, num_channels=3, width=224, height=224) 

	# pass the blob through the network and obtain the face detections
	faceNet.setInput(blob)
	detections = faceNet.forward()

	# initialize our list of faces, their corresponding locations, and the list of predictions from our face mask network
	faces = []
	locs = []
	preds = []

	# loop over the detections
	for i in range(0, detections.shape[2]):
		# extract the confidence (i.e., probability) associated with the detection
		confidence = detections[0, 0, i, 2]

		# filter out weak detections by ensuring the confidence is greater than the minimum confidence
		if confidence > 0.5:
			# compute the (x, y)-coordinates of the bounding box for the object
			box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
			(startX, startY, endX, endY) = box.astype("int")

			# ensure the bounding boxes fall within the dimensions of the frame
			(startX, startY) = (max(0, startX), max(0, startY))
			(endX, endY) = (min(w - 1, endX), min(h - 1, endY))

			# extract the face ROI, convert it from BGR to RGB channel ordering, resize it to 224x224, and preprocess it
			face = frame[startY:endY, startX:endX]
			face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
			face = cv2.resize(face, (224, 224))
			face = img_to_array(face)
			face = preprocess_input(face)

			# add the face and bounding boxes to their respective lists
			faces.append(face)
			locs.append((startX, startY, endX, endY))

	# only make a predictions if at least one face was detected
	if len(faces) > 0:
		# for faster inference we'll make batch predictions on *all*
		# faces at the same time rather than one-by-one predictions
		# in the above `for` loop
		faces = np.array(faces, dtype = "float32")
		preds = maskNet.predict(faces, batch_size = 32)	


	# return a 2-tuple of the face locations and their corresponding prediction
	return (locs, preds)

# load our serialized face detector model from disk
prototxtPath = r"face_detector\deploy.prototxt"
weightsPath = r"face_detector\res10_300x300_ssd_iter_140000.caffemodel"
faceNet = cv2.dnn.readNet(prototxtPath, weightsPath)

# load the face mask detector model from disk
maskNet = load_model("mask_detector.model")

# initialize the video stream
print("Starting video stream...")
vs = VideoStream(0).start()


def email_alert():
    sender_email = "email.alerts.baruch.2020@gmail.com"
    sender_name = 'Mask Detector ALERT'
    password = "vxmfahsxniaxbznq"
    receiver_emails = ['wangyde2013@gmail.com', 'yan.wang8@baruchmail.cuny.edu']
    receiver_names = ['Yan', 'Friend']
    email_body = '''No Mask Alert!\nThe system detected a person without mask.'''

    for receiver_email, receiver_name in zip(receiver_emails, receiver_names):
        print("Sending the email...")
        msg = MIMEMultipart()
        msg['To'] = formataddr((receiver_name, receiver_email))
        msg['From'] = formataddr((sender_name, sender_email))
        msg['Subject'] = 'Hello, ' + receiver_name + '. No Mask Alert!'     
        msg.attach(MIMEText(email_body, 'html'))

        try:
            # Open PDF file in binary mode
            with open(filename, 'rb') as attachment:
                            part = MIMEBase("application", "octet-stream")
                            part.set_payload(attachment.read())

            # Encode file in ASCII characters to send by email
            encoders.encode_base64(part)

            # Add header as key/value pair to attachment part
            part.add_header(
                    "Content-Disposition",
                    f"attachment; filename= {filename}",
            )

            msg.attach(part)
        except Exception as e:
                print(f'Oh no! We didn\'t found the attachment!\n{e}')
                break

        try:
                # Creating a SMTP session | use 587 with TLS, 465 SSL and 25
                server = smtplib.SMTP('smtp.gmail.com', 587)
                # Encrypts the email
                context = ssl.create_default_context()
                server.starttls(context=context)
                # We log in into our Google account
                server.login(sender_email, password)
                # Sending email from sender, to receiver with the email body
                server.sendmail(sender_email, receiver_email, msg.as_string())
                print('Email sent!')
        except Exception as e:
                print(f'Oh no! Something bad happened!\n{e}')
                break
        finally:
                print('Closing the server...')
                server.quit()



# loop over the frames from the video stream
while True:
	# grab the frame from the threaded video stream and resize it to 800 pixels
	frame = vs.read()
	frame = imutils.resize(frame, width = 800)

	# detect faces in the frame and determine if they are wearing a
	# face mask or not
	(locs, preds) = detect_and_predict_mask(frame, faceNet, maskNet)

	# loop over the detected face locations and their corresponding
	# locations
	for (box, pred) in zip(locs, preds):
		# unpack the bounding box and predictions
		(startX, startY, endX, endY) = box
		(mask, withoutMask) = pred

		# determine the class label and color we'll use to draw the bounding box and text
		if mask > withoutMask:
			label = "Mask" 
			color = (0, 255, 0)
		else:
			label = "No Mask" 
			color = (0, 0, 255)
			filename = pyautogui.screenshot()
			filename = cv2.cvtColor(np.array(filename), cv2.COLOR_RGB2BGR)
			cv2.imwrite("Screenshot.png", filename)
			filename = 'Screenshot.png'

			email_alert()
			
		#label = "Mask" if mask > withoutMask else "No Mask" 
		#color = (0, 255, 0) if label == "Mask" else (0, 0, 255)

		# include the probability in the label
		label = "{}: {:.2f}%".format(label, max(mask, withoutMask) * 100)

		# display the label and bounding box rectangle on the output frame
		cv2.putText(frame, label, (startX, startY - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.95, color, 2)
		cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)

	# show the output frame
	cv2.imshow("Frame", frame)
	key = cv2.waitKey(1) & 0xFF

	# if the `q` key was pressed, break from the loop
	if key == ord("q"):
		break
vs.stop()
cv2.destroyAllWindows()



