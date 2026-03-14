import os
import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader, random_split

# image transform
transform = transforms.Compose([
    transforms.Resize((128,128)),
    transforms.ToTensor()
])

# dataset class
class BUSIDataset(Dataset):

    def __init__(self, root):
        self.images = []
        self.masks = []

        for folder in ["benign","malignant"]:
            path = os.path.join(root,folder)

            for file in os.listdir(path):

                if "_mask" not in file:
                    img_path = os.path.join(path,file)

                    mask_name = file.replace(".png","_mask.png")
                    mask_path = os.path.join(path,mask_name)

                    if os.path.exists(mask_path):
                        self.images.append(img_path)
                        self.masks.append(mask_path)

    def __len__(self):
        return len(self.images)

    def __getitem__(self,idx):

        image = Image.open(self.images[idx]).convert("RGB")
        mask = Image.open(self.masks[idx]).convert("L")

        image = transform(image)
        mask = transform(mask)

        mask = (mask > 0).float()

        return image,mask


dataset = BUSIDataset("Dataset_BUSI_with_GT")

train_size = int(0.8*len(dataset))
test_size = len(dataset)-train_size

train_dataset,test_dataset = random_split(dataset,[train_size,test_size])

train_loader = DataLoader(train_dataset,batch_size=8,shuffle=True)
test_loader = DataLoader(test_dataset,batch_size=8)


# UNet
class DoubleConv(nn.Module):

    def __init__(self,in_c,out_c):
        super().__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(in_c,out_c,3,padding=1),
            nn.ReLU(),
            nn.Conv2d(out_c,out_c,3,padding=1),
            nn.ReLU()
        )

    def forward(self,x):
        return self.conv(x)


class UNet(nn.Module):

    def __init__(self):
        super().__init__()

        self.down1 = DoubleConv(3,64)
        self.pool1 = nn.MaxPool2d(2)

        self.down2 = DoubleConv(64,128)
        self.pool2 = nn.MaxPool2d(2)

        self.bottleneck = DoubleConv(128,256)

        self.up1 = nn.ConvTranspose2d(256,128,2,stride=2)
        self.conv1 = DoubleConv(256,128)

        self.up2 = nn.ConvTranspose2d(128,64,2,stride=2)
        self.conv2 = DoubleConv(128,64)

        self.final = nn.Conv2d(64,1,1)

    def forward(self,x):

        d1 = self.down1(x)
        p1 = self.pool1(d1)

        d2 = self.down2(p1)
        p2 = self.pool2(d2)

        bn = self.bottleneck(p2)

        u1 = self.up1(bn)
        m1 = torch.cat([u1,d2],dim=1)
        c1 = self.conv1(m1)

        u2 = self.up2(c1)
        m2 = torch.cat([u2,d1],dim=1)
        c2 = self.conv2(m2)

        return torch.sigmoid(self.final(c2))


model = UNet()

criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(),lr=0.001)

# training
for epoch in range(10):

    for images,masks in train_loader:

        outputs = model(images)

        loss = criterion(outputs,masks)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print("Epoch",epoch+1,"finished")


# accuracy
correct = 0
total = 0

with torch.no_grad():

    for images,masks in test_loader:

        outputs = model(images)

        preds = (outputs>0.5).float()

        correct += (preds==masks).sum().item()
        total += torch.numel(preds)

accuracy = correct/total*100

print("Accuracy:",accuracy)