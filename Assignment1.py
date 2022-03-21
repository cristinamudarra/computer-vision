"""
Created on Fri Mar  4 11:17:38 2022

@author: Cristina del Pilar Mudarra Pradas
Student Number: r0874660
[COMPUTER VISION] - ASSIGNMENT 1
"""

"""Skeleton code for python script to process a video using OpenCV package

:copyright: (c) 2021, Joeri Nicolaes
:license: BSD license
"""

""" python Assignment1.py -i 
C:/Users/mudar/Documents/KU Leuven/2ND SEMESTER/Computer Vision/Assignments/Assignment1/Videos/Video1.mp4
-o
C:/Users/mudar/Documents/KU Leuven/2ND SEMESTER/Computer Vision/Assignments/Assignment1/Videos/Output.mp4

"""

import cv2 as cv
import numpy as np
import sys
import imutils
import argparse



# helper function to change what you do based on video seconds
def between(cap, lower: int, upper: int) -> bool:
    return lower <= int(cap.get(cv.CAP_PROP_POS_MSEC)) < upper


def main(input_video_file: str, output_video_file: str) -> None:
    # OpenCV video objects to work with
    cap = cv.VideoCapture(input_video_file)
    fps = int(round(cap.get(5)))
    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))
    fourcc = cv.VideoWriter_fourcc(*'mp4v')        # saving output video as .mp4
    out = cv.VideoWriter(output_video_file, fourcc, fps, (frame_width, frame_height))

    # While loop to process each frame
    while cap.isOpened():
        ret, frame = cap.read()
        subtitle = ''
        
        if ret:
            if cv.waitKey(28) & 0xFF == ord('q'):
                break
            # Switch between color and grayscale
            if between(cap, 500, 4000):
                subtitle = 'Switch between color and grayscale'
                if between(cap, 1000, 2000) or between(cap, 3000, 4000):
                    frame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
                    
            # Gaussian filter with kernel 9
            if between(cap, 4000, 6000):
                subtitle = 'Gaussian filter with kernel (11, 11)'
                frame = cv.GaussianBlur(frame,(11,11),cv.BORDER_DEFAULT)
            
            # Gaussian filter with kernel 5
            if between(cap, 6000, 8000):
                subtitle = 'Gaussian filter with kernel (5, 5)'
                frame = cv.GaussianBlur(frame,(5,5),cv.BORDER_DEFAULT)
            
            # Bilateral filter with parameters
            if between(cap, 8000, 10000):
                subtitle = 'Bilateral filter with parameters (25, 250, 250)'
                frame = cv.bilateralFilter(frame, 25, 275, 275, cv.BORDER_DEFAULT)
            
            # Bilateral filter with parameters
            if between(cap, 10000, 12000):
                subtitle = 'Bilateral filter with parameters (5, 100, 100)'
                frame = cv.bilateralFilter(frame, 5, 100, 100, cv.BORDER_DEFAULT)
            
            # Thresholding of red color in BGR Color Space
            if between(cap, 12000, 15000):
                subtitle = 'Thresholding of red color in BGR Color Space'
                frame = cv.inRange(frame, (0, 0, 128), (71, 99, 255))
                
            # Thresholding of red color
            if between(cap, 15000, 18000):
                subtitle = 'Thresholding of red color in HSV Space'
                frame = cv.cvtColor(frame, cv.COLOR_BGR2HSV)
                frame = cv.inRange(frame, (0,120,120), (18, 250, 250) )
            
            # Thresholding of red color with dilate + erode
            if between(cap, 18000, 21000):
                subtitle = 'Thresholding of red color with dilate + erode in HSV Space'
                frame = cv.cvtColor(frame, cv.COLOR_BGR2HSV)
                frame = cv.inRange(frame, (0,120,120), (17, 255, 255))
                
                kernel = np.ones((7, 7), 'uint8')
                
                
                frame = cv.dilate(frame, kernel, iterations=1)
                frame = cv.erode(frame, kernel, iterations=1)
               
            
            # Sobel - edge detector -> Vertical lines
            if between(cap, 21000, 23000):
                subtitle = 'Sobel - Edge detector -> Vertical lines'
                frame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
                frame = cv.Sobel(frame, cv.CV_8U, dx=1, dy=0, ksize=3)
                frame = cv.cvtColor(frame, cv.COLOR_GRAY2BGR)
                frame = cv.cvtColor(frame, cv.COLOR_BGR2HSV)
                
            # Sobel - edge detector -> Horizontal lines   
            if between(cap, 23000, 25000):
                subtitle = 'Sobel - Edge detector -> Horizontal lines'
                frame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
                frame = cv.Sobel(frame, cv.CV_8U, dx=0, dy=1, ksize=3)
                frame = cv.cvtColor(frame, cv.COLOR_GRAY2BGR)
                frame = cv.cvtColor(frame, cv.COLOR_BGR2HSV)
                
            # Hough transform -> Detect circles with tweaked parameters
            if between(cap, 25000, 30000):
                subtitle = 'Hough circles with appropiate parameters'
                mask = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
                minDist_1 = 70 # If it is too small, multiple neighbor circles may be falsely detected in addition to a true one. If it is too large, some circles may be missed.
                param1_1 = 25 # Parameter of Canny, usually initialized to 200
                param2_1 = 25 # Usually 100. Setting it higher means more false negatives, lower more false positives.
                minRadius_1 = 2
                maxRadius_1 = 25 #10
                
                circles_1 = cv.HoughCircles(mask, cv.HOUGH_GRADIENT, 1, minDist_1, param1=param1_1, param2=param2_1, minRadius=minRadius_1, maxRadius=maxRadius_1)
            
                if circles_1 is not None:
                    circles_1 = np.uint16(np.around(circles_1))
                    for i in circles_1[0,:]:
                        cv.circle(frame, (i[0], i[1]), i[2], (0, 255, 0), 2)
                
                minDist_2 = 70 # If it is too small, multiple neighbor circles may be falsely detected in addition to a true one. If it is too large, some circles may be missed.
                param1_2 = 25 # Parameter of Canny, usually initialized to 200
                param2_2 = 25 # Usually 100. Setting it higher means more false negatives, lower more false positives.
                minRadius_2 = 25
                maxRadius_2 = 50 #10
                
                circles_2 = cv.HoughCircles(mask, cv.HOUGH_GRADIENT, 1, minDist_2, param1=param1_2, param2=param2_2, minRadius=minRadius_2, maxRadius=maxRadius_2)
                
                if circles_2 is not None:
                    circles_2 = np.uint16(np.around(circles_2))
                    for i in circles_2[0,:]:
                        cv.circle(frame, (i[0], i[1]), i[2], (0, 255, 0), 2)
                
        
            # Hough transform -> Detect circles with inappropiate parameters
            if between(cap, 30000, 35000):
                subtitle = 'Hough circles with inappropiate parameters'
                mask = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
                minDist = 150
                param1 = 55 
                param2 = 20 # smaller value-> more false circles
                minRadius = 3
                maxRadius = 80 
                circles = cv.HoughCircles(mask, cv.HOUGH_GRADIENT, 1, minDist, param1=param1, param2=param2, minRadius=minRadius, maxRadius=maxRadius)

                if circles is not None:
                    circles = np.uint16(np.around(circles))
                    for i in circles[0,:]:
                        cv.circle(frame, (i[0], i[1]), i[2], (0, 255, 0), 2)
                
                   
            # Flashy rectangle around object of interest
            if between(cap, 37000, 39000):
                
                subtitle = 'Flashy rectangle around object of interest'
                
                img_interest = cv.imread("Object.jpg", 0) # Image of object of interest in grayscale
                frame_gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY) 
                
                w,h = img_interest.shape[::-1]

                # Match with squares differences
                temp = cv.matchTemplate(frame_gray, img_interest, cv.TM_SQDIFF)
                
                (minVal, maxVal, minLoc, maxLoc) = cv.minMaxLoc(temp)              
                (startX, startY) = minLoc 
                
                endX = startX + img_interest.shape[1]
                endY = startY + img_interest.shape[0]               
                cv.rectangle(frame, (startX, startY), (endX, endY), (0, 0, 255), 2)
    
            
            # Likelihood of the object of interest in a position
            if between(cap, 39000, 42000):
                subtitle = 'Likelihood of the object of interest in a position'
                img_interest = cv.imread("Object.jpg", 0) # Image of the object of interest in grayscale
                frame_gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)               
                w,h = img_interest.shape[::-1]
                temp = cv.matchTemplate(frame_gray, img_interest, cv.TM_SQDIFF)
            
                inv_probability = cv.normalize(temp, dst=None, alpha=0, beta=255, norm_type=cv.NORM_MINMAX, dtype=cv.CV_8U)
                probability = cv.bitwise_not(inv_probability) 
                probability = cv.resize(probability, (frame.shape[1], frame.shape[0])) 
                frame = cv.cvtColor(probability, cv.COLOR_GRAY2BGR)
                
            
            # Detect and follow the tomato
            if between(cap, 42000, 48000):
                subtitle = 'Detect & follow the tomato'
                mask = cv.cvtColor(frame, cv.COLOR_BGR2HSV)
                mask = cv.inRange(mask, (0, 220, 20), (17, 255, 255))
                col = 250
                row = 150
                width = 120
                height = 300
                kernel = np.ones((9, 9), 'uint8')
                
                cnts = cv.findContours(mask.copy(), cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
                cnts = imutils.grab_contours(cnts)
                center = None
                #Only proceed if at least one contour was found
                if len(cnts) > 0:
                    # find the largest contour in the mask, then use
                    # it to compute the minimum enclosing circle and
                    # centroid
                    c = max(cnts, key=cv.contourArea)
                    ((x, y), radius) = cv.minEnclosingCircle(c)
                    M = cv.moments(c)
                    # only proceed if the radius meets a minimum size
                    if radius > 10:
                        center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))
                        # draw the circle on the frame,
                        cv.circle(frame, center, 5, (0, 0, 255), -1)
                
                
            # Change the color of the lemon
            if between(cap, 48000, 55000):
                if between(cap, 51000, 55000):
                    frame = cv.rotate(frame, cv.ROTATE_180)
                subtitle = 'Change color of the lemon +  invariant to rotation'
                mask = cv.cvtColor(frame, cv.COLOR_BGR2HSV)
                mask = cv.inRange(mask, (20, 150, 20), (50, 255, 255))
                
                kernel = np.ones((9, 9), 'uint8')
                
                mask = cv.erode(mask, kernel, iterations=1)
                mask = cv.dilate(mask, kernel, iterations=1)
                
                mask = cv.dilate(mask, kernel, iterations=1)
                mask = cv.erode(mask, kernel, iterations=1)
                frame[mask>0] = (80, 120, 255)  
                
                
            # Detection of the face
            if between(cap, 55000, 60000):
                subtitle = 'Face detection'
                face_cascade = cv.CascadeClassifier("C:/Users/mudar/Downloads/opencv-master/opencv-master/data/haarcascades/haarcascade_frontalface_default.xml")
                mask = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
                faces = face_cascade.detectMultiScale(mask, 1.3, 5)
                for (x,y,w,h) in faces:
                    img = cv.rectangle(frame, (x,y), (x+w,y+h), (255,0,0), 2)
                    roi_gray = frame[y:y+h, x:x+w]
                    roi_color = frame[y:y+h, x:x+w] 
                
                
                
            # Output subtitles
            cv.putText(frame, subtitle, (140, 440), cv.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

            # write frame that you processed to output
            out.write(frame)

            # (optional) display the resulting frame
            cv.imshow('Frame', frame)

            # Press Q on keyboard to  exit
            if cv.waitKey(25) & 0xFF == ord('q'):
                break

        # Break the loop
        else:
            break

    # When everything done, release the video capture and writing object
    cap.release()
    out.release()
    # Closes all the frames
    cv.destroyAllWindows()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='OpenCV video processing')
    parser.add_argument('-i', "--input", help='full path to input video that will be processed')
    parser.add_argument('-o', "--output", help='full path for saving processed video output')
    args = parser.parse_args()

    if args.input is None or args.output is None:
        sys.exit("Please provide path to input and output video files! See --help")

    main(args.input, args.output)





























