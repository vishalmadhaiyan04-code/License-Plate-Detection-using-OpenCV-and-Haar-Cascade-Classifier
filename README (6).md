# License Plate Detection using OpenCV and Haar Cascade Classifier
### Name: Janagiraman M
### Register Number: 212224230101
## Aim
To implement a License Plate Detection system using OpenCV and Haar Cascade Classifier, draw bounding boxes, crop the detected region, and blur the license plate to improve privacy. The detection accuracy is improved by tuning Haar Cascade parameters.

## Software Used
Python 3.7 or above  
OpenCV (opencv-python)  
NumPy  
Matplotlib  
Jupyter Notebook (Anaconda)  
Haar Cascade File: haarcascade_russian_plate_number.xml

## Algorithm
Import necessary libraries such as OpenCV and Matplotlib  
Read the input vehicle image  
Convert the original image to grayscale for faster computation  
Load the Haar Cascade classifier for license plate detection  
Detect license plate using detectMultiScale function  
Draw rectangle around detected area  
Crop the detected region using numpy slicing with (x, y, w, h) values  
Apply median blurring on the cropped region  
Replace the original region with blurred version  
Display final result using Matplotlib

## Program
```python
import cv2
import matplotlib.pyplot as plt

def display(img):
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    plt.figure(figsize=(10,6))
    plt.imshow(img_rgb)
    plt.axis('off')

img = cv2.imread("DATA/car_plate.jpg")
plate_cascade = cv2.CascadeClassifier("DATA/haarcascades/haarcascade_russian_plate_number.xml")

def detect_and_blur_plate(img):
    img_copy = img.copy()
    gray = cv2.cvtColor(img_copy, cv2.COLOR_BGR2GRAY)
    plates = plate_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4)

    for (x, y, w, h) in plates:
        roi = img_copy[y:y+h, x:x+w]
        blurred_roi = cv2.medianBlur(roi, 15)
        img_copy[y:y+h, x:x+w] = blurred_roi

    return img_copy

result = detect_and_blur_plate(img)
display(result)
```

## Output
<br>
<img width="800" height="400" alt="Screenshot 2025-11-22 233020" src="https://github.com/user-attachments/assets/97f14cae-797d-4771-b071-21b713a68ef7" />
<br>
<img width="800" height="400" alt="Screenshot 2025-11-22 233031" src="https://github.com/user-attachments/assets/4964d7b6-8b70-4b57-97bd-c632af58fa73" />
<br>
<img width="800" height="400" alt="Screenshot 2025-11-22 233046" src="https://github.com/user-attachments/assets/e509829f-0456-4ded-ae98-720b984edf56" />

## Modification Done

Parameter tuning was performed by adjusting scaleFactor and minNeighbors values in detectMultiScale to improve accuracy and reduce false detections. Median blur was applied to protect license plate information.

## Result

The License Plate Detection system was successfully implemented using OpenCV and Haar Cascade. The detected license plate region was blurred using median filtering. The modified values improved overall detection performance and output quality.

## Conclusion

This workshop demonstrates how classical computer vision methods like Haar Cascades can be used for real-time applications such as automated toll systems, smart parking, and traffic surveillance. Proper preprocessing and parameter tuning significantly improve detection results.



