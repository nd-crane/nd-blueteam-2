import cv2
import argparse
import os
import datetime
import threading
import queue
import yaml
import signal
import sys
import time
import numpy as np

def generate_long_timestamp():
    now = datetime.datetime.now()
    timestamp = now.strftime("%Y-%m-%d_%H-%M-%S-%f")[:-3]  # -f gives microseconds, and [:-3] trims it to milliseconds
    # timestamp = now.strftime("%Y-%m-%d_%H-%M-%S-%f")  # -f gives microseconds
    return timestamp

def resize_frame(frame, width, height):
    return cv2.resize(frame, (width, height))

def generate_timestamp():
    now = datetime.datetime.now()
    timestamp = now.strftime("%Y-%m-%d_%H-%M-%S")
    return timestamp

def capture_frames(rtsp_url, label, output_folder, frame_save_width, frame_save_height, frame_display_width, frame_display_height, file_format, frames_queue, exit_event, time_limit, start_barrier):
    # Wait for all threads to reach the start barrier before capturing frames
    start_barrier.wait()

    # Create a VideoCapture object for the RTSP stream
    cap = cv2.VideoCapture(rtsp_url)

    # Check if the VideoCapture object was successfully opened
    if not cap.isOpened():
        print(f"Error: Could not open RTSP stream {label}.")
        return

    # Create a subfolder for the stream using the label
    stream_folder = os.path.join(output_folder, label)
    if not os.path.exists(stream_folder):
        os.makedirs(stream_folder)

    frame_count = 0
    start_time = time.time()
    while True:
        if exit_event.is_set():
            break  # Exit the loop if the exit event is set

        ret, frame = cap.read()  # Read a frame from the stream
        frame_timestamp = generate_long_timestamp()
        
        if not ret:
            print(f"Error: Failed to read frame from RTSP stream {label}.")
            break

        # Resize the frame for saving
        if frame_save_width is not None and frame_save_height is not None:
            frame = resize_frame(frame, frame_save_width, frame_save_height)

        # Save the frame to the respective stream folder
        frame_count += 1
        frame_filename = os.path.join(stream_folder, f'frame_{frame_count:04d}__ts_{frame_timestamp}.{file_format}')
        cv2.imwrite(frame_filename, frame)

        # Resize the frame for displaying
        if frame_display_width is not None and frame_display_height is not None:
            frame = resize_frame(frame, frame_display_width, frame_display_height)

        # Put the frame in the queue for visualization
        frames_queue.put((label, frame))

        # Check if the time limit has been reached
        elapsed_time = time.time() - start_time
        if time_limit is not None and elapsed_time >= time_limit:
            exit_event.set()  # Set the exit event to terminate the thread

    # Release the VideoCapture object
    cap.release()

def signal_handler(signal, frame, exit_event):
    print("Received signal to terminate. Exiting...")
    exit_event.set()
    
# Function to calculate the grid layout based on combined frame dimensions
def calculate_grid_layout(num_streams, max_win_width, frame_width, frame_height):
    
    grid_cols = num_streams
        
    if frame_width * num_streams > max_win_width:
        grid_cols = int(max_win_width / frame_width)
        
    grid_rows = int(num_streams / grid_cols)
    if (num_streams % grid_cols):
         grid_rows += 1

    win_height = int(frame_height * grid_rows)
    win_width = int(frame_width * grid_cols)
    
    # Calculate the positions of the frames within the grid
    positions = [(i % grid_cols * frame_width, i // grid_cols * frame_height) for i in range(num_streams)]

    return win_width, win_height, positions

def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='Read and display video from multiple RTSP streams using OpenCV')
    parser.add_argument('--streams', required=True, help='YAML configuration file')
    parser.add_argument('--output_folder', default='recordings', help='Main output folder to save frames')
    parser.add_argument('--frame_save_width', type=int, default=960, help='Width to save frames (optional)')
    parser.add_argument('--frame_save_height', type=int, default=720, help='Height to save frames (optional)')
    parser.add_argument('--file_format', default='png', help='File format for saving frames (e.g., jpg, png)')
    parser.add_argument('--time_limit', type=int, default=30, help='Time limit in seconds to capture frames (optional)')
    parser.add_argument('--start_barrier', type=int, default=5, help='Delay in seconds before capturing frames (optional)')
    parser.add_argument('--max_screen_width', type=int, default=2400, help='Width of the display screen')
    parser.add_argument('--frame_display_width', type=int, default=624, help='Width to display frames (optional)')
    parser.add_argument('--frame_display_height', type=int, default=468, help='Height to display frames (optional)')
   
    args = parser.parse_args()

    # Read stream configuration from YAML file
    with open(args.streams, 'r') as streams:
        config = yaml.safe_load(streams)

    # Create the main output folder if it doesn't exist
    if not os.path.exists(args.output_folder):
        try:
            os.makedirs(args.output_folder)
        except OSError:
            print(f"Error: Could not create main output folder '{args.output_folder}'.")
            exit()
            
    # Create inside the main output folder another folder with name based on the current date and time
    timestamp = generate_timestamp()
    output_folder = os.path.join(args.output_folder, timestamp)
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)       

    # Create a thread for each RTSP stream
    threads = []
    frames_queue = queue.Queue()
    exit_event = threading.Event()
    start_barrier = threading.Barrier(len(config['streams']) + 1)  # Add one for the main thread

    for stream_config in config['streams']:
        label = stream_config['label']
        rtsp_url = stream_config['url']
        thread = threading.Thread(target=capture_frames, args=(rtsp_url, label, output_folder, args.frame_save_width,
                                                              args.frame_save_height, args.frame_display_width, args.frame_display_height,
                                                              args.file_format, frames_queue, exit_event, args.time_limit, start_barrier))
        thread.start()
        threads.append(thread)

    # Set up a signal handler to terminate threads on Ctrl+C
    signal.signal(signal.SIGINT, lambda signal, frame: signal_handler(signal, frame, exit_event))

    # Wait for all threads to reach the start barrier
    start_barrier.wait()

    # Calculate grid layout
    num_streams = len(config['streams'])
    win_width, win_height, positions = calculate_grid_layout(num_streams, args.max_screen_width, args.frame_display_width, args.frame_display_height)

    # Create the combined frame with the calculated dimensions
    combined_frame = np.zeros((win_height, win_width, 3), dtype=np.uint8)
    while not exit_event.is_set():
        while not frames_queue.empty():
            label, frame = frames_queue.get()
            frame_height, frame_width, _ = frame.shape
            position = positions[config['streams'].index(next(stream for stream in config['streams'] if stream['label'] == label))]
            combined_frame[position[1]:position[1] + frame_height, position[0]:position[0] + frame_width, :] = frame

        cv2.imshow('RTSP Streams', combined_frame)

        # Press 'q' to exit the loop
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release OpenCV window and wait for threads to complete
    cv2.destroyAllWindows()
    for thread in threads:
        thread.join()

if __name__ == "__main__":
    main()