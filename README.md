# Notre Dame Blue Team project for the 2023 RAITE exercise

## Nix

```bash
nix shell '.#devShells.x86_64-linux.default' -c python evaluate.py
```

## Notre Dame Team 2 Models - "SINATRA"
These models make use of the TorchVision implementaion of the Faster-RCNN architecture with a ResNet50 backbone to perform object detection. Further information on the model architecture may be found here: https://pytorch.org/vision/main/models/generated/torchvision.models.detection.fasterrcnn_resnet50_fpn_v2.html#torchvision.models.detection.fasterrcnn_resnet50_fpn_v2

During inference, the models take inputs of tensors with the form (B,C,H,W) where B is the batch dimension, C is the number of color channels, H is the height, and W is the width, and they output a list containing a dictionary of bounding boxes, predicted labels, and confidence scores for each image in the batch. Inputs are expected to contain values within the range [0,1], and the models were trained using RGB images that were 960x720 (width by height). The models were trained using only two classes, person and background, and therefore a predicted label of 1 indicates that a given bounding box contains a human subject. Each bounding box is in the form [x1,y1,x2,y2]. 
