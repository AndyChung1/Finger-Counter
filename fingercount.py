import cv2 as cv
import numpy as np
from sklearn.metrics import pairwise

background = None
accumulated_weight = 0.4

def calc_accumulated_avg(frame, accum_weight): # taking the average of the frame and the background variable
    global background
    
    if background is None:
        background = frame.copy().astype('float')
        return None
    
    cv.accumulateWeighted(frame, background, accum_weight)

def segment(frame, threshold = 25): # segmenting the hand using thresholding
    global background
    diff = cv.absdiff(background.astype('uint8'), frame) # taking the absolute difference between the background and frame
    
    ret, thresholded = cv.threshold(diff, threshold, 255, cv.THRESH_BINARY) # threshold the image to create binary frame

    # grab contours from the binary frame
    image, contours, hierarchy = cv.findContours(thresholded.copy(), cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE) 
    
    if len(contours) == 0: # return none if we get no contours
        return None
    else:
        hand_contour = max(contours, key = cv.contourArea) # the largest external contour should be your hand
        
        return (thresholded, hand_contour)

def count_fingers(thresholded, hand_contour):
    conv_hull = cv.convexHull(hand_contour)
    
    top = tuple(conv_hull[conv_hull[:,:,1].argmin()][0]) # grabbing the most extreme points in the polygon
    bottom = tuple(conv_hull[conv_hull[:,:,1].argmax()][0])
    left = tuple(conv_hull[conv_hull[:,:,0].argmin()][0])
    right = tuple(conv_hull[conv_hull[:,:,0].argmax()][0])
    
    center_x = (left[0] + right[0]) // 2 # determine center point
    center_y = (bottom[1] + top[1]) // 2
    
    # distances between center point and extreme points
    distance = pairwise.euclidean_distances([(center_x, center_y)], Y=[left, right, bottom, top])[0] 
    
    # create a circle that is a little smaller than the extreme point lengths
    max_distance = distance.max()
    
    circle_radius = int(0.7 * max_distance)
    
    circle_roi = np.zeros(thresholded.shape[:2], dtype = 'uint8') # create black area where the image is segmented 
    
    cv.circle(circle_roi, (center_x, center_y), circle_radius, 255, 10) # draw circle on the black frame
    
    # ANDing the circle and your hand which can easily show the number of fingers being held up
    circle_roi = cv.bitwise_and(thresholded, thresholded, mask = circle_roi) 
    cv.imshow('roi', circle_roi)
    
    # the contours should be the sections where the circle intersects with your hand
    image, contours, hierarchy = cv.findContours(circle_roi.copy(), cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)
    
    finger_count = 0
    
    for contour in contours:
        (x,y,w,h) = cv.boundingRect(contour)
        
        not_wrist = ((center_y + (0.4*center_y)) > (y + h))
        close_to_circle = (2*circle_radius*np.pi*0.25) > contour.shape[0]
        
        if not_wrist and close_to_circle:
            finger_count += 1
            
    return finger_count

cap = cv.VideoCapture(0)
num_frames = 0
roi_top = 20
roi_bottom = 300
roi_left = 300
roi_right = 600

while True:
    ret, frame = cap.read()
    frame = cv.flip(frame, 1)
    frame_copy = frame.copy()
    roi = frame[roi_top:roi_bottom, roi_left:roi_right] # take portion of the frame as roi
    
    gray = cv.cvtColor(roi, cv.COLOR_BGR2GRAY)
    gray = cv.GaussianBlur(gray, (7,7), 0)
    
    if num_frames < 60:
        calc_accumulated_avg(gray, accumulated_weight)
        
        if num_frames <= 59:
            cv.putText(frame_copy, 'getting background', (10,10), cv.FONT_HERSHEY_COMPLEX, 1, (0,0,255), 1)
#             cv.imshow('fingers', frame_copy)
    else:
        hand = segment(gray)
        
        if hand is not None:
            thresholded, hand_segment = hand
            cv.drawContours(frame_copy, [hand_segment + (roi_left,roi_top)], -1, (255,0,0), 5)
            fingers = count_fingers(thresholded, hand_segment)
            cv.putText(frame_copy, str(fingers), (70,50), cv.FONT_HERSHEY_COMPLEX, 1, (0,0,255), 1)
            cv.imshow('thresholded', thresholded)
            
    cv.rectangle(frame_copy, (roi_left, roi_top), (roi_right, roi_bottom), (0,0,255), 1)
    
    num_frames += 1
    
    cv.imshow('finger count', frame_copy)
    
    if (cv.waitKey(1) & 0xFF) == 27:
        break

cap.release()        
cv.destroyAllWindows()