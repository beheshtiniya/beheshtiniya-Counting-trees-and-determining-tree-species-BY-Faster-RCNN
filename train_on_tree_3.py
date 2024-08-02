import os
import pandas as pd
import torch
import torchvision
from PIL import Image
import matplotlib
matplotlib.use('TkAgg')
from torch.utils import data
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from torchvision.transforms import functional as F
CUDA_LAUNCH_BLOCKING = 1.
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print ("device:", device)
image_and_targets=[]


class trDataset(torch.utils.data.Dataset):
    def __init__(self, root, phase):
        self.root = root
        self.phase = phase

        # self.imgs = os.listdir(os.path.join(root, 'images'))
        self.targets = pd.read_csv(os.path.join(root, '{}_labels.csv'.format(phase)))
        self.imgs = self.targets['filename']

    def __getitem__(self, idx):
        img_path = os.path.join(self.root, 'images', self.imgs[idx])
        img = Image.open(img_path).convert('RGB')
        img = F.to_tensor(img)
        # print("idx:",idx)
        #
        box_list = self.targets[self.targets['filename'] == self.imgs[idx]]
        box_list = box_list[['xmin', 'ymin', 'xmax', 'ymax']].values
        boxes = torch.tensor(box_list, dtype=torch.float32)
        #
        # labels = torch.ones((len(box_list),), dtype=torch.int64)
        # print("labels",labels)
        label_list = self.targets[self.targets['filename'] == self.imgs[idx]]
        label_list = label_list[['class']].values.squeeze(1)
        labels = torch.tensor(label_list, dtype=torch.int64)
        # print("labels", labels)
        #
        target = {}
        target['boxes'] = boxes
        target['labels'] = labels
        #
        return img, target

    def __len__(self):
        return len(self.imgs)
train_dataset = trDataset('./dataset', 'train')


test_dataset = trDataset('./dataset', 'test')
print("************************************")
print(train_dataset.__getitem__(10))
print("************************************")
def new_concat(batch):
  return tuple(zip(*batch))
train_loader = torch.utils.data.DataLoader(train_dataset,
                            batch_size=2,
                            shuffle=True,
                            collate_fn=new_concat)

test_loader = torch.utils.data.DataLoader(test_dataset,
                            batch_size=1,
                            shuffle=True,
                            collate_fn=new_concat)
print("************************************")
print(next(iter(train_loader)))
print("************************************")
model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
model.roi_heads.box_predictor = torchvision.models.detection.faster_rcnn.FastRCNNPredictor(1024, 5)
model.to(device)
optimizer = torch.optim.SGD(model.parameters(), lr=0.005,
                            momentum=0.9, weight_decay=0.0005)

lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                               step_size=3,
                                               gamma=0.1)
import math
def train_one_epoch(model, optimizer, train_dataloader):
    model.train()
    total_loss = 0
    for images, targets in train_dataloader:
        images = [image.to(device) for image in images]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())
        total_loss += losses
        optimizer.zero_grad()
        losses.backward()
        optimizer.step()
    return total_loss/len(train_dataloader)
num_epochs = 50

for epoch in range(num_epochs):
    print("start train")
    loss = train_one_epoch(model, optimizer, train_loader)
    print('epoch [{}]:  \t lr: {}  \t loss: {}  '.format(epoch, lr_scheduler.get_last_lr(), loss))
    lr_scheduler.step()

torch.save(model.state_dict(), "./checkpoints/best_model.pth")
