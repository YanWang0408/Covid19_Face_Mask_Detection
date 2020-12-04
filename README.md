# Covid19_Face_Mask_Detection


> :memo: This is a very basic Convolutional Neural Network (CNN) model using TensorFlow with Keras library and OpenCV to detect if you are wearing a face mask. 


### Data Source: 
1. Flickr-Faces-HQ3 (FFHQ), FFHQ contains 70000 high-quality images of human faces in PNG images at 1024 × 1024 resolution and is free of use. The FFHQ dataset offers a lot of variety in terms of age, ethnicity, viewpoint, lighting, and image background. https://github.com/NVlabs/ffhq-dataset. 

2. MaskedFace-Net - a dataset of human faces with a correctly or incorrectly worn mask (137,016 images) based on the FFHQ,  by Adnane Cabani, Karim Hammoudi, Halim Benhabiles, and Mahmoud Melkemi. https://arxiv.org/pdf/2008.08016.pdf.

### Codes: 

#### Step 1: Training_Mask_Detector.ipynb

1. Import libraries (**install tensorflow on 64-bit Python**)
2. Data preprocessing
    - Download with mask and without mask images from data source.
    - Transfer multiple dataset formats (jpg, jepg, png) into jpg in batch.
    - Identify invalid data and remove them.
    - Normalize data size. From 1024 by 1024 into 224 by 224.
3. Transfer image into numpy array
   ```
      image = load_img(img_path, target_size = (224, 224))
      image = img_to_array(image)
      image = preprocess_input(image)
    ```
4. Building and Training the model
   - tensorflow.keras.applications.MobileNetV2 
   > MobileNetV2 is a convolutional neural network architecture that seeks to perform well on mobile devices. It is based on an inverted residual structure where the residual connections are between the bottleneck layers.
   - Batch size = 32, epochs = 20
   - Model accuary: Traning set 0.99, Testing set 0.99.
5. Serialize the model to disk
   ```
      model.save("mask_detector.model", save_format="h5")
    ```

#### Step 2: Realtime_mask_detection.py

1. Import libraries
2. Define detect_and_predict_mask funtion, it requires three parameters:
   - Image frame captured from video
   - Face detection model
   - Mask detection model
3. Initialize the video stream
4. Destroy All Windows


### Step 3: Additional steps - Alerts
1. waston_text2speech.py

    Customize your alert message into audio, with multiple mainstream language choices.
2. Realtime_mask_detection_audio_alerts.py

    Once the system detects people without a mask, the alert will sound.
3. email_alert_with_screenshot.py

    Customize your email alert message.
4. Realtime_mask_detection_email_alerts.py
    

### Credits::pushpin: 

[Deep Learning Tutorials](https://www.udemy.com/course/machinelearning/learn/lecture/6761138#overview) 

[OpenCV Python Tutorial For Beginners](https://www.youtube.com/watch?v=eX7wXfNLFDw&list=PLS1QulWo1RIa7D1O6skqDQ-JZ1GGHKK-K&index=18)

[Deep learning: How OpenCV’s blobFromImage works](https://www.pyimagesearch.com/2017/11/06/deep-learning-opencvs-blobfromimage-works/)

[COVID-19: Face Mask Detection using TensorFlow and OpenCV](https://towardsdatascience.com/covid-19-face-mask-detection-using-tensorflow-and-opencv-702dd833515b)

[Face Mask Detection using Python, Keras, OpenCV and MobileNet | Detect masks real-time video streams](https://www.youtube.com/watch?v=Ax6P93r32KU&t=918s)

