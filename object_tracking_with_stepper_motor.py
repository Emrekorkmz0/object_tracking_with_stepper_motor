import cv2
import numpy as np
#import RPi.GPIO as GPIO
import time
import math
from ultralytics import YOLO
import supervision as sv

# GPIO pin definitions for Motor X and Motor Y
# Motor X controls the X-axis
IN1_X, IN2_X, IN3_X = 17, 18, 27
# Motor Y controls the Y-axis
IN1_Y, IN2_Y, IN3_Y = 23, 24, 25


# GPIO setup
GPIO.setmode(GPIO.BCM)
GPIO.setup([IN1_X, IN2_X, IN3_X, IN1_Y, IN2_Y, IN3_Y], GPIO.OUT)

# Step motor sequence (same for both motors)
step_sequence = [
    [1, 0, 0, 1],
    [1, 0, 0, 0],
    [1, 1, 0, 0],
    [0, 1, 0, 0],
    [0, 1, 1, 0],
    [0, 0, 1, 0],
    [0, 0, 1, 1],
    [0, 0, 0, 1]
]

x1,y1,x2,y2=0,0,0,0

x_current, y_current = 0, 0  # Initial position


# Function to rotate both motors in parallel for given steps
def rotate_motors_in_parallel(steps_x, steps_y, direction_x, direction_y, delay=0.001):
    max_steps = max(steps_x, steps_y)

    # Iterate up to the maximum number of steps required
    for step in range(max_steps):
        # Determine the step index for each motor
        step_index_x = (step % len(step_sequence)) * direction_x
        """
        #buraya bir if koşulu koyup steps_y ye kadar dönder loopu step_index_y için
        if step < steps_y:
            step_index_y = (step % len(step_sequence)) * direction_y
            step_y = step_sequence[step_index_y % len(step_sequence)]
            
        """
        step_index_y = (step % len(step_sequence)) * direction_y

        # Get the step sequence for each motor
        step_x = step_sequence[step_index_x % len(step_sequence)]
        step_y = step_sequence[step_index_y % len(step_sequence)]

        # Set the GPIO output for Motor X
        GPIO.output(IN1_X, step_x[0])
        GPIO.output(IN2_X, step_x[1])
        GPIO.output(IN3_X, step_x[2])

        # Set the GPIO output for Motor Y
        GPIO.output(IN1_Y, step_y[0])
        GPIO.output(IN2_Y, step_y[1])
        GPIO.output(IN3_Y, step_y[2])

        time.sleep(delay)

# Calculate steps required for each axis
def calculate_steps(target, current, steps_per_revolution=4096):
    delta = target - current
    angle = math.degrees(math.atan2(delta, 1))  # Simplified as delta directly affects rotation
    steps = int(steps_per_revolution * abs(angle) / 360)  # Convert angle to steps
    direction = 1 if delta >= 0 else -1
    return steps, direction



def detection_for_target(detection):
    global x1,x2,y1,y2
    try:
        confidence_rate = round(detection[2], 1) 
        if confidence_rate >0.2:
        
            x1, y1, x2, y2 = detection[0][0], detection[0][1], detection[0][2], detection[0][3]
            return  x1, y1, x2, y2   
    except Exception as e:
        confidence_rate = 0
        return None

    
    return None
def main():
    cap=cv2.VideoCapture(0)
    model=YOLO("FPW.pt")

    #box_annotater = sv.BoundingBoxAnnotator()
    box_annotater=sv.BoxAnnotator(
        thickness=2,
        text_thickness=2,
        text_scale=1)  
    while True:
        _,frame=cap.read()
        
        result=model(frame)[0]
        detections=sv.Detections.from_ultralytics(result)
        #print("Tracker id:",detections)
        frame=box_annotater.annotate(scene=frame,
                                     detections=detections
                                     #labels=labels
                                     )
        #if you need thread dont use loop make it out of loop
        try:
            for detection in detections:
                print("Detection:",detection)
                detection_for_target(detection)
                x_target=int(x1+x2)/2
                y_target=int(y1+y2)/2
                # Calculate steps and direction for each axis
                steps_x, direction_x = calculate_steps(x_target, x_current)
                steps_y, direction_y = calculate_steps(y_target, y_current)
                #currentleri değiştirmeyi unuttuk burayı kotrol et
                x_current=x_target
                y_current=y_target
                rotate_motors_in_parallel(steps_x,steps_y,direction_x,direction_y)
        except Exception as e:
            pass

        cv2.imshow("Frame",frame)
    
        if cv2.waitKey(10)==27:
            break

if __name__=="__main__":
    main()
