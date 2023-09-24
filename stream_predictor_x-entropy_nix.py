from threading import Thread
import cv2
import argparse
import time
import torch
import torchvision.models as models
import torchvision.transforms as transforms
from torchvision.ops import nms as nms


FONT = cv2.FONT_HERSHEY_SIMPLEX

def write_to_csv(frame_number, boxes, scores, csv_path, score_threshold=0.0, timestamp=0.0):    
    with open(csv_path, 'a') as csvfile:
        for box, score in zip(boxes, scores):
            if score > score_threshold:
                x1, y1, x2, y2 = map(int, box)
                vx = x2 - x1
                vy = y2 - y1
                line = f"{frame_number},[{x1},{vx},{y1},{vy}];{score:.3f},{timestamp}\n"
                csvfile.write(line) 

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

class StreamInput(Thread):
    _stream: cv2.VideoCapture
    _should_exit: bool = False

    # Latest Information
    _latest_timestamp = 0.0

    def __init__(self, location: str, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self._latest_frame = None

        # Initialize Stream
        self._stream = cv2.VideoCapture(location) 
        if not self._stream.isOpened():
            raise Exception("can't open video writer")

        # Reduce buffer size if supported
        self._stream.set(cv2.CAP_PROP_BUFFERSIZE, 3)

    def isOpened(self) -> bool:
        return self._stream.isOpened()
    
    def width(self) -> int:
        return int(self._stream.get(cv2.CAP_PROP_FRAME_WIDTH))
    
    def height(self) -> int:
        return int(self._stream.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    def latest(self):
        if self._latest_frame is None:
            return (None, None)
        return (self._latest_timestamp, self._latest_frame.copy())

    def run(self) -> None:
        while not self._should_exit:
            ret, frame = self._stream.read()
            if not ret:
                break
            self._latest_frame = frame
            self._latest_timestamp = self._stream.get(cv2.CAP_PROP_POS_MSEC)

        self._stream.close()
        self._stream.release()

    def stop(self):
        self._should_exit = True

class RTSPOutput(Thread):
    def __init__(self, width: int, height: int, fps: int, location: str, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self._should_exit = False
        self._latest_frame = None
        self._timestep = 1.0 / fps
        self._stream = cv2.VideoWriter('appsrc ! videoconvert' + \
            ' ! video/x-raw,format=I420' + \
            ' ! x264enc speed-preset=ultrafast key-int-max=' + str(fps * 2) + \
            ' ! video/x-h264,profile=baseline' + \
            f' ! rtspclientsink protocols=tcp location={location}',
            cv2.CAP_GSTREAMER, 0, fps, (width, height), True)
        if not self._stream.isOpened():
            raise Exception("can't open video writer")
    
    def update(self, frame):
        self._latest_frame = frame.copy()
        
    def run(self) -> None:
        while not self._should_exit:
            start = time.time()

            if self._latest_frame is not None:
                self._stream.write(self._latest_frame)
            
            diff = time.time() - start
            if diff > self._timestep:
                time.sleep(diff)

        self._stream.release()
        print("Output finished")

    def stop(self):
        self._should_exit = True

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

def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='Read and display video from multiple RTSP streams using OpenCV')
    parser.add_argument('--input_rtsp', required=True, help='Input Stream RTSP URL')
    parser.add_argument('--output_rtsp', required=True, help='Output Stream RTSP URL')
    parser.add_argument("--weights_path", required=True, help="Path to the model weights file")
    parser.add_argument('--output_fname', required=True, help='Name of the output file with predictions')
    parser.add_argument('--score_threshold', type=float, default=0.0, help='score_threshold')

    args = parser.parse_args()

    # load model    
    device = torch.device('cuda')
    weights_path = args.weights_path
    model = models.detection.fasterrcnn_resnet50_fpn_v2(num_classes=2,trainable_backbone_layers=5)
    model.load_state_dict(torch.load(weights_path)['state_dict'])
    model = model.to(device)
    model.eval()
    
    time.sleep(2)
    
    # NEW CODE BY TIM
    predictor = prediction_helper()
    
    # Open the stream 
    cap_time = time.time()

    input_rtsp = StreamInput(args.input_rtsp)
    input_rtsp.start()

    output_rtsp = RTSPOutput(input_rtsp.width(), input_rtsp.height(), 30, args.output_rtsp)
    output_rtsp.start()

    print("cap time", time.time()-cap_time)

    frame_count = 0
    last_timestamp = 0.0

    while input_rtsp.isOpened():

        frame_time = time.time()
        timestamp, frame = input_rtsp.latest()
        
        if frame is None or last_timestamp == timestamp:
            time.sleep(0.01)
            continue

        last_timestamp = timestamp
        frame_count += 1

        # Make prediction
        boxes, scores = predictor(frame,model,device,iou_threshold=0.25)

        # Write predictions for RAITE ouput format 
        write_to_csv(frame_count, boxes, scores, args.output_fname, timestamp)
        
        # Display the frame with boxes
        draw_boxes(frame, boxes, scores, score_threshold=0.9)

        output_rtsp.update(frame)

        print("frame",frame_count , "frame time", time.time()-frame_time, " timestamp", timestamp)

        
    input_rtsp.stop()
    input_rtsp.join()
    
    output_rtsp.stop()
    output_rtsp.join()

if __name__ == "__main__":
    main()