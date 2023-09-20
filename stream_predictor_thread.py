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
import torchvision.transforms as transforms
import csv 

# Define the desired resolution
frame_display_width = 640
frame_display_height = 480


def write_to_csv(frame_number, boxes, scores, csv_path, score_threshold=0.0):    
    with open(csv_path, 'a') as csvfile:
        for box, score in zip(boxes, scores):
            if score > score_threshold:
                x1, y1, x2, y2 = map(int, box)
                vx = x2 - x1
                vy = y2 - y1
                line = f"{frame_number},[{x1},{vx},{y1},{vy}];{score:.3f}\n"
                csvfile.write(line)

def save_to_csv(data_queue, csv_path, score_threshold=0.0):
    with open(csv_path, 'a', newline='') as csvfile:
        writer = csv.writer(csvfile)
        
        while True:
            data = data_queue.get()
            
            if data is None:  # Sentinel value to exit the loop
                break

            frame_number, boxes, scores = data
            for box, score in zip(boxes, scores):
                if score > score_threshold:
                    x1, y1, x2, y2 = map(int, box)
                    vx = x2 - x1
                    vy = y2 - y1
                    line = [frame_number, f"[{x1},{vx},{y1},{vy}]", f"{score:.3f}"]
                    writer.writerow(line)


# This function will run in a separate thread and will keep displaying frames
def display_frame(frame_queue):
    while True:
        data = frame_queue.get()
        if data is None:  # Sentinel value to exit the loop
            break

        frame, boxes, scores = data        
        draw_boxes(frame, boxes, scores)

        # Resize the frame
        #frame = cv2.resize(frame, (frame_display_width, frame_display_height))
        
        # Display
        # # TODO: refactory frame name 
        cv2.imshow('ND BlueTeam 2 - Model 1', frame)

        # Exit when 'q' is pressed
        if cv2.waitKey(1) == ord('q'):
            break


def make_prediction2(image_array, model, device):
    # Define the image transformation to be applied to the input NumPy array
    transform = transforms.Compose([
        transforms.ToPILImage(),  # Convert to PIL image
        transforms.ToTensor(),    # Convert to a PyTorch tensor
        #transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    # Apply the transformation to the input NumPy array
    image = transform(image_array).to(device)

    # Add a batch dimension (assuming model expects a batch of images)
    image = torch.unsqueeze(image, dim=0)

    # Make predictions
    predictions = model(image)
    boxes = predictions[0]['boxes'].detach().cpu().numpy()
    scores = predictions[0]['scores'].detach().cpu().numpy() 

    return boxes, scores

def impath_make_prediction(image_path, model, device):
    
    image = torch.unsqueeze(torchvision.io.read_image(image_path).to(torch.float32)/255., dim=0).to(device)
    predictions = model(image)       
    boxes = predictions[0]['boxes'].detach().cpu().numpy()
    scores = predictions[0]['scores'].detach().cpu().numpy() 
    return boxes, scores

def draw_boxes(image, boxes, scores, score_threshold=0.0):
    
    # Iterate over each box and draw it on the image
    for box, score in zip(boxes, scores):
        if score > score_threshold:  # Only draw boxes with scores higher than the threshold
            # Convert box coordinates to integers
            x1, y1, x2, y2 = map(int, box)

            # Draw the bounding box
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)

            # Display the score above the bounding box
            cv2.putText(image, f"{score:.2f}", (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

def initialize_csv_with_headers(output_path):
    headers = ["#Frame Number", "Box (x, vx, y, vy)", "Score"]
    with open(output_path, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(headers)

def main():
     # Parse command-line arguments
    parser = argparse.ArgumentParser(description='Read and display video from multiple RTSP streams using OpenCV')
    parser.add_argument('--stream_label', default="Stream_0", help='Stream Label')
    parser.add_argument('--stream_url', required=True, help='Stream RTSP URL')
    parser.add_argument("--weights_path", required=True, help="Path to the model weights file")
    parser.add_argument('--output_folder', default='output', help='Main output folder to save results')
    parser.add_argument('--frame_save_width', type=int, default=960, help='Width to save frames (optional)')
    parser.add_argument('--frame_save_height', type=int, default=720, help='Height to save frames (optional)')
    parser.add_argument('--file_format', default='png', help='File format for saving frames (e.g., jpg, png)')
    parser.add_argument('--time_limit', type=int, default=30, help='Time limit in seconds to capture frames (optional)')
    parser.add_argument('--score_threshold', type=float, default=0.0, help='score_threshold')

    args = parser.parse_args()

    #stream_label = 'Stream_0'
    #stream_url = 'rtsp://admin:Marialufy2@192.168.0.101:65534'
    #stream_url = 'rtsp://192.168.0.103:554/H264Video'
    #stream_url = '/home/pmoreira/tai-raite/raite-stream_recorder/video_1.mkv'

    stream_folder = os.path.join(args.output_folder, args.stream_label)
    if not os.path.exists(stream_folder):
        os.makedirs(stream_folder)

    # TODO: refactory this    
    raite_output_path = os.path.join(stream_folder, 'ndblueteam_2-model_1.csv')   

    # load model
    device = torch.device('cuda')
    weights_path = args.weights_path
    model = models.detection.fasterrcnn_resnet50_fpn_v2(num_classes=2,trainable_backbone_layers=5)
    model.load_state_dict(torch.load(weights_path)['state_dict'])
    model = model.to(device)
    model.eval()
    
    #_ =  impath_make_prediction("sample_img.png", model, device)

    # Create a flag to signal the threads to exit
    exit_flag = threading.Event()

    # Function to handle exit
    def exit_program():
        print("Exiting program...")
        exit_flag.set()

    # Register a signal handler for Ctrl+C
    signal.signal(signal.SIGINT, lambda signum, frame: exit_program())
    
    frame_queue = queue.Queue(maxsize=1)
    data_queue = queue.Queue()

    initialize_csv_with_headers(raite_output_path)

    # Start the saving thread
    saving_thread = threading.Thread(target=save_to_csv, args=(data_queue, raite_output_path))
    saving_thread.start()
    
    # Start the display thread
    display_thread = threading.Thread(target=display_frame, args=(frame_queue,))
    display_thread.start()
    
    cap_time = time.time()    
    cap = cv2.VideoCapture(args.stream_url)

    if not cap.isOpened():
        print("Error: Could not open stream.")
        exit()

    print("cap time", time.time()-cap_time)

    frame_count = 0
    
    while True:

        frame_time = time.time()

        ret, frame = cap.read()
        if not ret:
            print("Error: Couldn't read frame.")
            break
                
        frame_count += 1

        # Make prediction        
        boxes, scores = make_prediction2(frame, model, device)

        # Put the frame into the queue to be displayed by a separate thread
        frame_queue.put((frame, boxes, scores))

        # Put the data into the queue to be saved by a separate thread
        data_queue.put((frame_count, boxes, scores)) 

        # Exit when the exit_flag is set with Ctrl+C 
        if exit_flag.is_set():
            break

        print("frame time", time.time()-frame_time)


    frame_queue.put(None)  # Sentinel value to indicate that we are done
    display_thread.join()
    
    data_queue.put(None)  # Sentinel value to signal saving thread to finish
    saving_thread.join()

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()