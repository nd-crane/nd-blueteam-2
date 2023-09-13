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
import torch
import torchvision.io
import torchvision.models as models


def write_to_csv(frame_number, predictions, csv_path, score_threshold=0.5):
    
    with open(csv_path, 'a') as csvfile:
        # Extract boxes, labels, and scores from predictions
        boxes = predictions[0]['boxes'].detach().cpu().numpy()
        scores = predictions[0]['scores'].detach().cpu().numpy()

        for box, score in zip(boxes, scores):
            if score > score_threshold:
                x1, y1, x2, y2 = map(int, box)
                vx = x2 - x1
                vy = y2 - y1
                line = f"{frame_number},[{x1},{vx},{y1},{vy}];{score:.3f}\n"
                csvfile.write(line)

def draw_boxes(image, predictions, output_path, score_threshold=0.5):

    # Extract boxes, labels, and scores from predictions
    boxes = predictions[0]['boxes'].detach().cpu().numpy()
    scores = predictions[0]['scores'].detach().cpu().numpy()

    # Iterate over each box and draw it on the image
    for box, score in zip(boxes, scores):
        if score > score_threshold:  # Only draw boxes with scores higher than the threshold
            # Convert box coordinates to integers
            x1, y1, x2, y2 = map(int, box)

            # Draw the bounding box
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)

            # Display the score above the bounding box
            cv2.putText(image, f"{score:.2f}", (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Save the image with bounding boxes
    cv2.imwrite(output_path, image)    
    

def make_prediction(image_path, model, device):
    image = torch.unsqueeze(torchvision.io.read_image(image_path).to(torch.float32)/255., dim=0).to(device)
    predictions = model(image)        
    return predictions

def resize_frame(frame, width, height):
    return cv2.resize(frame, (width, height))

def generate_timestamp():
    now = datetime.datetime.now()
    timestamp = now.strftime("%Y-%m-%d_%H-%M-%S")
    return timestamp

def generate_long_timestamp():
    now = datetime.datetime.now()
    timestamp = now.strftime("%Y-%m-%d_%H-%M-%S-%f")[:-3]  # -f gives microseconds, and [:-3] trims it to milliseconds
    # timestamp = now.strftime("%Y-%m-%d_%H-%M-%S-%f")  # -f gives microseconds
    return timestamp

def capture_frames(rtsp_url, label, output_folder, frame_save_width, frame_save_height, frame_display_width, frame_display_height, file_format, frames_queue, exit_event, time_limit, start_barrier, model, device):
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
    
    original_frames_folder = os.path.join(stream_folder, 'orig_frames')
    if not os.path.exists(original_frames_folder):
        os.makedirs(original_frames_folder)
    
    result_frames_folder = os.path.join(stream_folder, 'result_frames')
    if not os.path.exists(result_frames_folder):
        os.makedirs(result_frames_folder)
        
    raite_output_path = os.path.join(stream_folder, 'ndblueteam_2-model_1.csv')        

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

        # Save original frames to the respective stream folder
        frame_count += 1
        orig_frame_fn = os.path.join(original_frames_folder, 
                                      f'frame_{frame_count:04d}__ts_{frame_timestamp}.{file_format}')
        cv2.imwrite(orig_frame_fn, frame)

        # Make prediction
        predictions =  make_prediction(orig_frame_fn, model, device)
        
        # Save prediction to frame image
        res_frames_fn = os.path.join(result_frames_folder, 
                                      f'frame_{frame_count:04d}__ts_{frame_timestamp}.{file_format}')
        draw_boxes(frame, predictions, res_frames_fn, score_threshold=0.5)         
         
        # Write predictions for RAITE ouput format 
        write_to_csv(frame_count, predictions, raite_output_path, score_threshold=0.5)            
         
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
    parser.add_argument('--output_folder', default='output', help='Main output folder to save results')
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
    #output_folder = os.path.join(args.output_folder, timestamp)
    output_folder = args.output_folder
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)       

    # load model    
    device = torch.device('cuda')
    weights_path = "models/best_model.pth"
    model = models.detection.fasterrcnn_resnet50_fpn_v2(num_classes=2,trainable_backbone_layers=5)
    model.load_state_dict(torch.load(weights_path)['state_dict'])
    model = model.to(device)
    model.eval()    
    
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
                                                              args.file_format, frames_queue, exit_event, args.time_limit, start_barrier,
                                                              model, device))
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