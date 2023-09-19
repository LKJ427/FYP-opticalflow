import cv2 as cv
import numpy as np
import csv
import math
import tkinter as tk
from tkinter import filedialog
import matplotlib.pyplot as plt

def select_video_file():
    root = tk.Tk()
    root.withdraw()  # Hide the main window
    file_path = filedialog.askopenfilename(title="Select a video file", filetypes=[("Video Files", "*.mp4")])
    return file_path

# Function to get user input for start and end times in seconds
def get_time_range():
    start_time = float(input("Enter the start time (in seconds): "))
    end_time = float(input("Enter the end time (in seconds, -1 for entire video): "))
    return start_time, end_time

# Video file path (now selected by the user)
video_path = select_video_file()

# Check if the user canceled file selection
if not video_path:
    print("No video file selected. Exiting.")
    exit()

# Get user input for the time range
start_time, end_time = get_time_range()

# Parameters for Shi-Tomasi corner detection
feature_params = dict(maxCorners=300, qualityLevel=0.2, minDistance=2, blockSize=7)
# Parameters for Lucas-Kanade optical flow
lk_params = dict(winSize=(15, 15), maxLevel=2, criteria=(cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 10, 0.03))

# Initialize video capture
cap = cv.VideoCapture(video_path)
color = (0, 255, 0)

# Set the frame position to the start time
start_frame = int(start_time * cap.get(cv.CAP_PROP_FPS))
cap.set(cv.CAP_PROP_POS_FRAMES, start_frame)

# Read the first frame
ret, first_frame = cap.read()
prev_gray = cv.cvtColor(first_frame, cv.COLOR_BGR2GRAY)  # Initialize prev_gray with the first frame
prev = cv.goodFeaturesToTrack(prev_gray, mask=None, **feature_params)
mask = np.zeros_like(first_frame)

# Create a CSV file for writing
csv_filename = "optical_flow_data.csv"
csv_file = open(csv_filename, 'w', newline='')
csv_writer = csv.writer(csv_file)
csv_writer.writerow(["Frame", "Point_X", "Point_Y", "Flow_X", "Flow_Y"])

# Lists to store aggregated flow vectors
flow_x_values = []
flow_y_values = []

# Function to convert angle to direction
def angle_to_direction(angle):
    if -45 <= angle < 45:
        return "East"
    elif 45 <= angle < 135:
        return "South"
    elif -135 <= angle < -45:
        return "North"
    else:
        return "West"

frame_count = 0

# Output video settings
fps = cap.get(cv.CAP_PROP_FPS)
output_video_path = "output_video.mp4"
fourcc = cv.VideoWriter_fourcc(*'mp4v')
w = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
h = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))
size = (w, h)
output_video = cv.VideoWriter(output_video_path, fourcc, fps, size, True)

# If the user inputs -1 for end_time, set end_time to the total duration of the video
if end_time == -1:
    end_time = cap.get(cv.CAP_PROP_FRAME_COUNT) / cap.get(cv.CAP_PROP_FPS)

while True:
    ret, frame = cap.read()

    # Check if we have reached the end of the specified time range
    if not ret or (frame_count / fps) >= end_time:
        break

    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    prev = cv.goodFeaturesToTrack(prev_gray, mask=None, **feature_params)
    next, status, error = cv.calcOpticalFlowPyrLK(prev_gray, gray, prev, None, **lk_params)
    good_old = prev[status == 1].astype(int)
    good_new = next[status == 1].astype(int)

    total_flow_x = 0
    total_flow_y = 0

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

    # Aggregate flow vectors
    flow_x_values.append(total_flow_x)
    flow_y_values.append(total_flow_y)

    average_flow_magnitude = math.sqrt(total_flow_x ** 2 + total_flow_y ** 2)
    average_flow_direction = math.degrees(math.atan2(total_flow_y, total_flow_x))
    average_direction_text = angle_to_direction(average_flow_direction)
    print(f"Frame {frame_count}: Avg Flow Magnitude = {average_flow_magnitude:.2f}, Avg Flow Direction = {average_direction_text}")

    avg_flow_text = f"Avg Magnitude: {average_flow_magnitude:.2f}, Avg Direction: {average_direction_text}"
    frame = cv.putText(frame, avg_flow_text, (10, 30), cv.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    output = cv.add(frame, mask)
    output_video.write(output)
    prev_gray = gray.copy()
    prev = good_new.reshape(-1, 1, 2)
    #cv.imshow("sparse optical flow", output)

    if cv.waitKey(10) & 0xFF == ord('q'):
        break

    frame_count += 1

# Close the CSV file and release resources
csv_file.close()
cap.release()
output_video.release()
cv.destroyAllWindows()

# Plot the aggregated flow vectors
plt.figure(figsize=(10, 6))
plt.quiver(0, 0, sum(flow_x_values), sum(flow_y_values), angles='xy', scale_units='xy', scale=1, color='blue')
plt.xlim(0, sum(flow_x_values))
plt.ylim(0, sum(flow_y_values))
plt.xlabel('X Direction')
plt.ylabel('Y Direction')
plt.title('Aggregated Optical Flow Direction')
plt.grid()
plt.show()
