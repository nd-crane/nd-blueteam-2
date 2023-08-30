import json,os,sys,csv,time,argparse
import numpy as np
import matplotlib.pyplot as plt
import torchvision.models as models
from torchvision.datasets import CocoDetection as CocoDataset
from torchvision.datasets import wrap_dataset_for_transforms_v2 as wrap_dataset
import torch
import torch.nn as nn
import torch.optim as optim
import RaiteDataset
import torchvision.transforms.v2 as transforms
import torchvision
from engine import train_one_epoch
sys.path.append("../")

def save_results(log,fieldnames,output_path):
    with open(os.path.join(output_path,"log.csv"),"w") as f:
        writer = csv.writer(f)
        limit = len(log[fieldnames[0]])
        for i in range(limit):
            writer.writerow([log[x][i] for x in fieldnames])
    return

def generate_masks(images,targets,invert=False):
    masks = []
    for image,target in zip(images,targets):
        mask = torch.ones_like(image)
        for box in target["boxes"]:
            bbox = [round(i.item()) for i in box]
            mask = torchvision.transforms.functional.erase(mask,i=bbox[0],j=bbox[1],w=bbox[2]-bbox[0],h=bbox[3]-bbox[1],v=0)
        if invert:
            mask = 1-mask
        masks.append(mask)
    return torch.stack(masks)

def PGD(model,x,targets,epsilon=0,attack_iters=16,step_size=1./255,masks=None,clamp_min=0.0,clamp_max=1.0):
    if masks is None:
        masks = torch.ones_like(x)
    adv_x = x + torch.mul(torch.empty_like(x).uniform_(-epsilon,epsilon),masks)
    adv_x = torch.clamp(adv_x,min=clamp_min,max=clamp_max)
    adv_x.requires_grad = True

    for _ in range(attack_iters):
        with torch.cuda.amp.autocast(enabled=False):
            loss_dict = model(adv_x,targets)
            total_loss = sum(loss for loss in loss_dict.values())
            gradients = torch.autograd.grad(total_loss,adv_x)[0]
            gradients = torch.mul(gradients,masks)
            adv_x = adv_x + step_size*gradients.sign()
            eta = torch.clamp(adv_x-x,min=-epsilon,max=epsilon)
            adv_x = torch.clamp(x+eta,min=clamp_min,max=clamp_max)
    return adv_x

# Description of all argument
parser = argparse.ArgumentParser()
parser.add_argument('-batchSize', type=int, default=32)
parser.add_argument('-nEpochs', type=int, default=50)
parser.add_argument('-data_root_path', default='/scratch365/tredgrav/Datasets/RAITE/dataset/frames',required=False,type=str)
parser.add_argument('-annotations_file', default='/scratch365/tredgrav/Datasets/RAITE/dataset/train_dataset.json', required=False,type=str)
parser.add_argument('-output_path', required=False,type=str)
parser.add_argument('-train_adversarially',action="store_true")
parser.add_argument('-use_mask',action="store_true")
parser.add_argument('-invert_mask',action="store_true")

args = parser.parse_args()
device = torch.device('cuda')

# Load the model
#model = models.detection.fasterrcnn_resnet50_fpn_v2(weights=models.detection.FasterRCNN_ResNet50_FPN_V2_Weights.COCO_V1,num_classes=2)
model = models.detection.fasterrcnn_resnet50_fpn_v2(num_classes=2,trainable_backbone_layers=5)
model = model.to(device)
params = [p for p in model.parameters() if p.requires_grad]

# Set up our data loader
data_root_path = args.data_root_path
train_path = args.annotations_file
training_dataset = RaiteDataset.RaiteDataset(data_root_path,train_path)
train_loader = torch.utils.data.DataLoader(training_dataset,batch_size=args.batchSize,shuffle=True,collate_fn = lambda x: tuple(zip(*x)))

# Optimizer settings (no hyper parameter tuning was performed)
solver = optim.SGD(params, lr=0.01, weight_decay=1e-6, momentum=0.9)
lr_sched = optim.lr_scheduler.StepLR(solver, step_size=10, gamma=0.1)

# Create destination folder
output_path = args.output_path
os.makedirs(output_path,exist_ok=True)
best_model_path = os.path.join(output_path,"best_model.pth")

# Train?
for epoch in range(args.nEpochs):
    model.train()
    for images, targets in train_loader:
        if args.use_mask:
            masks = generate_masks(images,targets,invert=args.invert_mask)
            masks = masks.to(device)
        else:
            masks = None
        images = torch.stack(list(image.to(device) for image in images))
        images = images.to(device)
        targets = [{k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in t.items()} for t in targets]
        if args.train_adversarially:
            images = PGD(model,images,targets,epsilon=8./255,masks=masks)
        with torch.cuda.amp.autocast(enabled=False):
            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())
        solver.zero_grad()
        losses.backward()
        solver.step()
    states = {
            "epoch":epoch + 1,
            "state_dict":model.state_dict(),
            "optimizer":solver.state_dict(),
            }
    torch.save(states,best_model_path)

    lr_sched.step()

states = {
        "epoch":epoch + 1,
        "state_dict":model.state_dict(),
        "optimizer":solver.state_dict(),
        }
torch.save(states,best_model_path)
