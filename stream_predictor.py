from typing import Any
import cv2
import argparse
import os
import datetime
import time
import numpy as np
import torch
import torchvision.io
import torchvision.models as models
import torchvision.transforms as transforms
from torchvision.ops import nms as nms


def write_to_csv(frame_number, boxes, scores, csv_path, score_threshold=0.0):    
    with open(csv_path, 'a') as csvfile:
        for box, score in zip(boxes, scores):
            if score > score_threshold:
                x1, y1, x2, y2 = map(int, box)
                vx = x2 - x1
                vy = y2 - y1
                line = f"{frame_number},[{x1},{vx},{y1},{vy}];{score:.3f}\n"
                csvfile.write(line)


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

class prediction_helper():
    def __init__(self):
        self.transform = transforms.Compose([
        # transforms.ToPILImage(),  # Convert to PIL image
        transforms.ToTensor(),    # Convert to a PyTorch tensor
        #transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
        
    def __call__(self, image_array,model,device,iou_threshold=None):

        # Apply the transformation to the input NumPy array
        image = cv2.cvtColor(image_array,cv2.COLOR_BGR2RGB)
        image = self.transform(image).to(device)
        # Add a batch dimension (assuming model expects a batch of images)
        image = torch.unsqueeze(image, dim=0)

        # Make predictions
        predictions = model(image)
        boxes = predictions[0]['boxes']
        scores = predictions[0]['scores']

        # Perform non-maximum surpression
        if iou_threshold is not None:
            nms_results = nms(boxes,scores,iou_threshold)
            boxes = boxes[nms_results]
            scores = scores[nms_results]
        boxes = boxes.detach().cpu().numpy()
        scores = scores.detach().cpu().numpy()

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
    parser.add_argument('--frame_display_width', type=int, default=640, help='Width to display frames (optional)')
    parser.add_argument('--frame_display_height', type=int, default=480, help='Height to display frames (optional)')
    parser.add_argument('--score_threshold', type=float, default=0.0, help='score_threshold')


    args = parser.parse_args()
   
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
    
    boxes, scores =  impath_make_prediction("sample_img.png", model, device)
    
    # NEW CODE BY TIM
    predictor = prediction_helper()

    # Open the stream
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

        # Resize frame
        #frame = cv2.resize(frame, (960, 720))
        
        # Make prediction
        # boxes, scores =  make_prediction2(frame, model, device)
        boxes, scores = predictor(frame,model,device,iou_threshold=0.25)        


        # Write predictions for RAITE ouput format 
        write_to_csv(frame_count, boxes, scores, raite_output_path)                    
        
        # Display the frame with boxes
        draw_boxes(frame, boxes, scores, score_threshold=0.)

        # Display
        # # TODO: refactory frame name 
        #frame = cv2.resize(frame, (960, 720))
        cv2.imshow('ND BlueTeam 2 - Model 1', frame)

        # Exit when 'q' is pressed
        if cv2.waitKey(1) == ord('q'):
            break

        print("frame time", time.time()-frame_time)
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()