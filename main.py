import numpy as np
import torch # type: ignore
import torch.nn as nn # type: ignore
import torch.optim as optim # type: ignore
import torch.nn.functional as F # type: ignore
import json
import os
import cv2 # type: ignore
import pandas as pd # type: ignore
from torchvision import transforms # type: ignore
from torch.utils.data import Dataset, DataLoader # type: ignore
from torchmetrics.classification import MulticlassPrecision, MulticlassRecall, MulticlassF1Score # type: ignore
from torch.optim.lr_scheduler import ReduceLROnPlateau # type: ignore
from torch.amp import autocast, GradScaler # type: ignore

"""Declare Constants"""
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]
BATCH_SIZE = 16
HEIGHT = 224
WIDTH = 224
NUM_EPOCHS = 20


def load_json(file_path: str) -> dict:
    with open(file_path, 'r') as f:
        return json.load(f)

def uniform_frame_sampling(video_path, num_frames):
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
       
    frame_indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)
    frames = []

    for i in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ret, frame = cap.read()
        if ret:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame)

    cap.release()
    return frames

class VideoDataset(Dataset):   
    def __init__(self, video_folder, json_data, transform=None, num_frames=40):
        self.video_folder = video_folder
        self.transform = transform
        self.videos = [f for f in os.listdir(video_folder)]
        self.json_data = {item['file']: item for item in json_data}
        self.labels = self.get_labels()
        self.num_frames = num_frames

    def get_labels(self):
        labels = []
        for f in self.videos:
            file_name = os.path.splitext(f)[0]
            label = self.json_data.get(file_name, {}).get('label')
            labels.append(label)
        return labels

    def __len__(self):
        return len(self.videos)
    
    def __getitem__(self, idx):
        video_path = os.path.join(self.video_folder, self.videos[idx])
        frames = uniform_frame_sampling(video_path, self.num_frames)
       
        if len(frames) == 0:
            return torch.zeros(3, self.num_frames, HEIGHT, WIDTH), -1  
        
        if len(frames) < self.num_frames:
            frames += frames[-1:] * (self.num_frames - len(frames)) 

        if self.transform:
            frames = [self.transform(frame) for frame in frames]
        
        frames = torch.stack(frames)  
        frames = frames.permute(1, 0, 2, 3)  
        label = self.labels[idx]
        
        return frames, label

class Transformations:
    def __init__(self, mean, std, height, width, train=True):
        self.mean = mean
        self.std = std
        self.height = height
        self.width = width
        self.train = train 
        self.transform = self.get_transforms()
    
    def get_transforms(self):
        if self.train:
            return transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize((self.height, self.width)),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(self.mean, self.std)
            ])
        else:
            return transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize((self.height, self.width)),
                transforms.ToTensor(),
                transforms.Normalize(self.mean, self.std)
            ])
        
    def __call__(self, frame):
        return self.transform(frame)

def main():
    
    train_video_folder = 'data/train_clipped'
    test_video_folder = 'data/test_clipped'
    val_video_folder = 'data/val_clipped'

    train_transform = Transformations(MEAN, STD, HEIGHT, WIDTH, train=True)
    test_transform = Transformations(MEAN, STD, HEIGHT, WIDTH, train=False)
    val_transform = Transformations(MEAN, STD, HEIGHT, WIDTH, train=False)

    train_json = load_json('MS-ASL/MSASL_train.json')
    test_json = load_json('MS-ASL/MSASL_test.json')
    val_json = load_json('MS-ASL/MSASL_val.json')

    train_dataset = VideoDataset(train_video_folder, train_json, transform=train_transform)
    test_dataset = VideoDataset(test_video_folder, test_json,  transform=test_transform)
    val_dataset = VideoDataset(val_video_folder, val_json,  transform=val_transform)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=8)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=8)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=8)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = torch.hub.load('facebookresearch/pytorchvideo', 'slowfast_r50', pretrained=True)
    model.blocks[6].proj = nn.Linear(model.blocks[6].proj.in_features, 1000)
    model.to(device)

    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.001, weight_decay=0.001, momentum=0.9)
    scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.1, patience=5, min_lr=1e-5)

    precision_metric = MulticlassPrecision(num_classes=1000, average='macro').to(device)
    recall_metric = MulticlassRecall(num_classes=1000, average='macro').to(device)
    f1_metric = MulticlassF1Score(num_classes=1000, average='macro').to(device)

    best_accuracy = 0.0
    os.makedirs('saved_models', exist_ok=True)
    MODEL_PATH = os.path.join('saved_models', 'slowfast_best_model.pth')

    for epoch in range(NUM_EPOCHS):
        print(f"Epoch {epoch+1}/{NUM_EPOCHS}")
  
        precision_metric.reset()
        recall_metric.reset()
        f1_metric.reset()

        model.train()

        running_loss = 0.0
        correct = 0
        total = 0
        scaler = GradScaler()
        for frames, labels in train_loader:
            if (labels == -1).any():
                print("Skipping batch")  
                continue

            frames = frames.to(device)
            labels = labels.to(device)

            slow_frames = frames[:, :, :8, :, :]  
            fast_frames = frames[:, :, 8:, :, :]

            slow_frames = slow_frames.to(device)
            fast_frames = fast_frames.to(device)  
            frames_input = [slow_frames, fast_frames]

            optimizer.zero_grad()

            with autocast(device_type = device.type):
                outputs = model(frames_input)
                loss = loss_fn(outputs, labels)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            _, predicted = torch.max(outputs.data, 1)
            running_loss += loss.item()
            correct +=(labels==predicted).sum().item()
            total += labels.size(0)
      
            precision_metric.update(predicted, labels)
            recall_metric.update(predicted, labels)
            f1_metric.update(predicted, labels)

        
        epoch_precision = precision_metric.compute().item()
        epoch_recall = recall_metric.compute().item()
        epoch_f1 = f1_metric.compute().item()

        epoch_loss = running_loss/len(train_loader)
        epoch_acc = 100.00* correct / total

        print(f"   - Training dataset. Accuracy: {epoch_acc:.3f}%, Precision: {epoch_precision:.3f}, Recall: {epoch_recall:.3f}, F1-Score: {epoch_f1:.3f}, Epoch loss: {epoch_loss:.3f}")

        val_loss = 0.0
        val_correct = 0
        val_total = 0
        model.eval()
        with torch.no_grad():
            correct = 0
            total = 0
            for frames, labels in val_loader:
                if (labels == -1).any():
                    print("Skipping batch")  
                    continue
                frames, labels = frames.to(device), labels.to(device)
                slow_frames = frames[:, :, :8, :, :]  
                fast_frames = frames[:, :, 8:, :, :]

                slow_frames = slow_frames.to(device)
                fast_frames = fast_frames.to(device)
                frames_input = [slow_frames, fast_frames]
                outputs = model(frames_input)
                loss = loss_fn(outputs, labels)
                val_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                val_correct += (predicted == labels).sum().item()
                val_total += labels.size(0)
        val_acc = 100.0 * val_correct / val_total    
        print(f"Validation Loss: {val_loss/len(val_loader):.4f}, Accuracy: {val_acc:.2f}%")

        if val_acc > best_accuracy:
            best_accuracy = val_acc
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'best_accuracy': best_accuracy
            }, MODEL_PATH)
            print(f"Model saved at epoch {epoch + 1} with validation accuracy: {val_acc:.2f}%")
    
    print("\nFinal Test Set Evaluation:")
    model.eval()
    test_correct, test_total = 0, 0
    with torch.no_grad():
        for frames, labels in test_loader:
            if (labels == -1).any():
                print("Skipping batch")  
                continue
            frames, labels = frames.to(device), labels.to(device)
            slow_frames, fast_frames = frames[:, :, :8, :, :], frames[:, :, 8:, :, :]
            frames_input = [slow_frames.to(device), fast_frames.to(device)]

            outputs = model(frames_input)
            _, predicted = outputs.max(1)
            test_correct += (predicted == labels).sum().item()
            test_total += labels.size(0)

    print(f"Test Accuracy: {100.0 * test_correct / test_total:.2f}%")


    
if __name__ == '__main__':
    main()
