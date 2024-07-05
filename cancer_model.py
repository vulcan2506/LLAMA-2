import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from tqdm import tqdm
from torchvision import datasets,models,transforms
from torch.utils.data import DataLoader


data_dir = "/home/vulcan/Downloads/Skin and Blood Cancer/archive(1)/melanoma_cancer_dataset"

data_transforms = {
   "train":transforms.Compose([
       transforms.RandomResizedCrop(224),
       transforms.RandomHorizontalFlip(),
       transforms.ToTensor(),
       transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
   ]),
    "test":transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
    ])
}

image_dataset = {x: datasets.ImageFolder(f"{data_dir}/{x}",data_transforms[x]) for x in ["train","test"] }
dataloaders = {x: torch.utils.data.DataLoader(image_dataset[x],batch_size=4,num_workers=0) for x in ["train","test"]}

model = models.resnet50(pretrained=True)
device = "cuda" if torch.cuda.is_available() else "cpu"
model = model.to(device)

num_filters = model.fc.in_features
model.fc = nn.Linear(num_filters,3)
criteria = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(),lr = 0.001)

train_loss_list = []
train_acc_list = []
val_lost_list = []
val_acc_list = []

num_epochs = 25

for epoch in tqdm(range(num_epochs)):
    for phase in ["train","test"]:
        if phase =="train":
            model.train()
        else:
            model.eval()
        running_loss = 0.0
        corrects = 0

        for inputs,labels in dataloaders[phase]:
            optimizer.zero_grad()
            inputs,labels = inputs.to(device), labels.to(device)
            
            with torch.set_grad_enabled(phase=="train"):
                outputs = model(inputs)
                loss = criteria(outputs,labels)
                
                if phase == "train":
                    loss.backward()
                    optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            _,preds = torch.max(outputs,1)
            corrects += torch.sum(preds==labels.data)

        epochs_loss = running_loss / len(image_dataset[phase])
        epoch_acc = corrects.double() / len(image_dataset[phase])
        print(f'{phase} Loss: {epochs_loss:.4f}, Acc:{epoch_acc:.4f}')

torch.save("model2.pt")