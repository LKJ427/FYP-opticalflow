import cv2 as cv
import numpy as np
import csv
import math
import time

# Parameters for Shi-Tomasi corner detection
feature_params = dict(maxCorners=300, qualityLevel=0.2, minDistance=2, blockSize=7)
# Parameters for Lucas-Kanade optical flow
lk_params = dict(winSize=(15, 15), maxLevel=2, criteria=(cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 10, 0.03))

# The video file path
video_path = "shibuya.mp4"

cap = cv.VideoCapture(video_path)
color = (0, 255, 0)
ret, first_frame = cap.read()
prev_gray = cv.cvtColor(first_frame, cv.COLOR_BGR2GRAY)
prev = cv.goodFeaturesToTrack(prev_gray, mask=None, **feature_params)
mask = np.zeros_like(first_frame)

# Create a CSV file for writing
csv_filename = "optical_flow_data.csv"
csv_file = open(csv_filename, 'w', newline='')
csv_writer = csv.writer(csv_file)
csv_writer.writerow(["Frame", "Point_X", "Point_Y", "Flow_X", "Flow_Y"])

def angle_to_direction(angle):
    if angle >= -45 and angle < 45:
        return "East"
    elif angle >= 45 and angle < 135:
        return "South"
    elif angle >= -135 and angle < -45:
        return "North"
    else:
        return "West"

frame_count = 0
start_time = time.time()
total_flow_x = 0
total_flow_y = 0

frame_interval = 1.0  # Time interval between frames in seconds
pixels_to_meters = 0.01  # Scaling factor (adjust according to your scene)

# Output video writer
output_video_path = "output_video.avi"
fourcc = cv.VideoWriter_fourcc(*'XVID')
output_video = cv.VideoWriter(output_video_path, fourcc, 30.0, (500, 500))  # Set resolution to 500x500

while cap.isOpened():
    ret, frame = cap.read()

    if frame is None:
        break

    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    prev = cv.goodFeaturesToTrack(prev_gray, mask=None, **feature_params)
    next, status, error = cv.calcOpticalFlowPyrLK(prev_gray, gray, prev, None, **lk_params)
    good_old = prev[status == 1].astype(int)
    good_new = next[status == 1].astype(int)

    for i, (new, old) in enumerate(zip(good_new, good_old)):
        a, b = new.ravel()
        c, d = old.ravel()
        mask = cv.line(mask, (a, b), (c, d), color, 2)
        frame = cv.circle(frame, (a, b), 3, color, -1)

        flow_x = a - c
        flow_y = b - d

        total_flow_x += flow_x
        total_flow_y += flow_y

        csv_writer.writerow([frame_count, a, b, flow_x, flow_y])

    elapsed_time = time.time() - start_time
    if elapsed_time >= frame_interval:
        average_flow_magnitude = math.sqrt(total_flow_x ** 2 + total_flow_y ** 2)
        average_speed = (average_flow_magnitude * pixels_to_meters) / frame_interval
        average_flow_direction = math.degrees(math.atan2(total_flow_y, total_flow_x))
        average_direction_text = angle_to_direction(average_flow_direction)
        print(f"Frame {frame_count}: Avg Speed = {average_speed:.2f} meters/second, Avg Direction = {average_direction_text}")

        avg_flow_text = f"Avg Speed: {average_speed:.2f} m/s, Avg Direction: {average_direction_text}"
        frame = cv.putText(frame, avg_flow_text, (10, 30), cv.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        start_time = time.time()
        total_flow_x = 0
        total_flow_y = 0

    output = cv.add(frame, mask)
    output_video.write(output)  # Write the frame to the output video without resizing
    prev_gray = gray.copy()
    prev = good_new.reshape(-1, 1, 2)
    cv.imshow("sparse optical flow", output)

    if cv.waitKey(10) & 0xFF == ord('q'):
        break

    frame_count += 1

# Release the output video writer
output_video.release()

# Close the CSV file and release resources
csv_file.close()
cap.release()
cv.destroyAllWindows()
