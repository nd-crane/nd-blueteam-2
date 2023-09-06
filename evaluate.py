import torch
import torchvision.io
import torchvision.models as models

device = torch.device('cuda')
model = models.detection.fasterrcnn_resnet50_fpn_v2(num_classes=2,trainable_backbone_layers=5)
weights_path = "best_model.pth"
model.load_state_dict(torch.load(weights_path)['state_dict'])
model = model.to(device)
model.eval()
image = torch.unsqueeze(torchvision.io.read_image("sample_img.png").to(torch.float32)/255., dim=0).to(device)
predictions = model(image)
print(predictions)