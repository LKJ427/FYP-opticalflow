import cv2
import numpy as np
import csv
import matplotlib.pyplot as plt

# Initialize video capture
cap = cv2.VideoCapture('shibuya.mp4')

# Create a VideoWriter object to save the output
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('output.avi', fourcc, 20.0, (640, 480))

# Lucas-Kanade parameters
lk_params = dict(winSize=(15, 15), maxLevel=2, criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

# Create some random colors for drawing optical flow
color = np.random.randint(0, 255, (100, 3))

# Read the first frame
ret, old_frame = cap.read()
old_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)

# Initialize variables for tracking speed and direction
average_speed_per_second = []
average_speed_per_minute = []
frame_count = 0
fps = cap.get(cv2.CAP_PROP_FPS)

# Initialize feature points for optical flow (p0)
# You can use a feature detection method (e.g., Shi-Tomasi or Harris) to initialize p0
p0 = cv2.goodFeaturesToTrack(old_gray, maxCorners=100, qualityLevel=0.3, minDistance=7, blockSize=7)

# Check if any feature points were found
if p0 is None or len(p0) == 0:
    print("No feature points found in the first frame.")
    exit()

# Ensure data type and shape of p0 are as expected
p0 = p0.astype(np.float32)
p0 = p0.reshape(-1, 1, 2)

# Initialize mask for drawing optical flow vectors and circles
mask = np.zeros_like(old_frame)

# Histogram for flow angles
angle_hist = {'North': [], 'South': [], 'East': [], 'West': []}

# CSV file for data logging
csv_filename = 'crowd_flow_data.csv'

# Initialize lists for plotting speed change
time_points = []
speed_points = []

# Create subplots
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

with open(csv_filename, mode='w', newline='') as csv_file:
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(['Time (s)', 'Average Speed (px/s)', 'Dominant Direction'])

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Calculate optical flow using Lucas-Kanade
        p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lk_params)

        # Select good points
        good_new = p1[st == 1]
        good_old = p0[st == 1]

        # Calculate the flow vectors and speeds
        flow_vectors = good_new - good_old
        speeds = np.linalg.norm(flow_vectors, axis=1)
        average_speed = np.mean(speeds)

        # Calculate the flow angles
        angles = np.arctan2(flow_vectors[:, 1], flow_vectors[:, 0])
        dominant_direction = ""

        for angle in angles:
            if -np.pi / 4 < angle <= np.pi / 4:
                dominant_direction = "East"
            elif np.pi / 4 < angle <= 3 * np.pi / 4:
                dominant_direction = "North"
            elif -3 * np.pi / 4 < angle <= -np.pi / 4:
                dominant_direction = "South"
            else:
                dominant_direction = "West"

            angle_hist[dominant_direction].append(angle)

        # Update frame count
        frame_count += 1

        # Append the speed to per-second and per-minute lists
        average_speed_per_second.append(average_speed)
        if frame_count % int(fps) == 0:
            average_speed_per_minute.append(np.mean(average_speed_per_second))
            # Calculate dominant direction based on the histogram
            dominant_direction = max(angle_hist, key=lambda k: len(angle_hist[k]))
            angle_hist = {key: [] for key in angle_hist}  # Reset histograms

            # Write data to CSV file
            elapsed_time = frame_count / fps
            csv_writer.writerow([elapsed_time, np.mean(average_speed_per_minute), dominant_direction])
            average_speed_per_second = []

            # Append data for speed change graph
            time_points.append(elapsed_time)
            speed_points.append(np.mean(average_speed_per_minute))

        # Draw optical flow vectors on the frame
        for i, (new, old) in enumerate(zip(good_new, good_old)):
            a, b = new.ravel().astype(int)  # Convert to integers
            c, d = old.ravel().astype(int)  # Convert to integers
            mask = cv2.line(mask, (a, b), (c, d), color[i].tolist(), 2)
            frame = cv2.circle(frame, (a, b), 5, color[i].tolist(), -1)

        # Combine the frame and the mask
        img = cv2.add(frame, mask)

        # Display the frame
        cv2.imshow('Crowd Flow', img)

        # Write the frame to the output video
        out.write(img)

        # Update the previous frame and points
        old_gray = frame_gray.copy()
        p0 = good_new.reshape(-1, 1, 2)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

# Release resources
cap.release()
out.release()
cv2.destroyAllWindows()

# Plot histograms for each direction
for direction, angles in angle_hist.items():
    ax1.hist(angles, bins=20, range=(-np.pi, np.pi), label=direction)

ax1.set_title("Flow Angle Histogram")
ax1.set_xlabel("Angle (radians)")
ax1.set_ylabel("Frequency")
ax1.legend()

# Plot the speed change graph
ax2.plot(time_points, speed_points)
ax2.set_title("Speed Change Over Time")
ax2.set_xlabel("Time (s)")
ax2.set_ylabel("Average Speed (px/s)")

# Show the plots
plt.show()
