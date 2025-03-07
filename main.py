import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import os
import string
import random
from tqdm import tqdm
import numpy as np
from torch.nn.utils.rnn import pad_sequence

CHARACTERS = string.ascii_lowercase + string.digits + string.ascii_uppercase
NUM_CLASSES = len(CHARACTERS)
BLANK_LABEL = NUM_CLASSES 
MAX_CAPTCHA_LENGTH = 4
IMG_WIDTH = 72
IMG_HEIGHT = 24
BATCH_SIZE = 128
EPOCHS = 100
LEARNING_RATE = 0.0005

class CaptchaDataset(Dataset):
    def __init__(self, data_dir, transform=None, augmentation=False):
        self.data_dir = data_dir
        self.transform = transform
        self.augmentation = transforms.Compose([
            transforms.RandomRotation(10),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
            transforms.GaussianBlur(kernel_size=3)
        ]) if augmentation else None
        
        self.image_files = [f for f in os.listdir(data_dir) 
                           if f.lower().endswith(('.png', '.jpg'))]
        if not self.image_files:
            raise ValueError(f"No images in {data_dir}")

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_name = self.image_files[idx]
        label_str = os.path.splitext(img_name)[0].lower()
        
        img_path = os.path.join(self.data_dir, img_name)
        image = Image.open(img_path).convert('L')
        
        if self.augmentation:
            image = self.augmentation(image)
        if self.transform:
            image = self.transform(image)
            
        label = [CHARACTERS.index(c) for c in label_str if c in CHARACTERS]
        return image, torch.tensor(label, dtype=torch.long)

transform = transforms.Compose([
    transforms.Resize((IMG_HEIGHT, IMG_WIDTH)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.456], std=[0.224])
])

train_dataset = CaptchaDataset('data/train', transform=transform, augmentation=True)
val_dataset = CaptchaDataset('data/val', transform=transform)

def collate_fn(batch):
    images = torch.stack([item[0] for item in batch])
    labels = [item[1] for item in batch]
    lengths = torch.tensor([len(lbl) for lbl in labels])
    
    padded_labels = pad_sequence(labels, batch_first=True, padding_value=BLANK_LABEL)
    return images, padded_labels, lengths

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, 
                         shuffle=True, num_workers=0,
                         collate_fn=collate_fn)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, 
                       shuffle=False, num_workers=0,
                       collate_fn=collate_fn)

class CaptchaCRNN(nn.Module):
    def __init__(self):
        super(CaptchaCRNN, self).__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(0.2),
            
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(0.2),
            
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d((2, 2)),
            nn.Dropout2d(0.2)
        )
        
        cnn_out_channels = 256
        cnn_out_height = IMG_HEIGHT // 8
        cnn_out_width = IMG_WIDTH // 8
        
        self.rnn = nn.GRU(cnn_out_channels * cnn_out_height, 
                         hidden_size=512, 
                         num_layers=3, 
                         bidirectional=True, 
                         batch_first=True,
                         dropout=0.3)
        self.fc = nn.Linear(1024, NUM_CLASSES + 1)

    def forward(self, x):
        cnn_features = self.cnn(x)
        batch, channels, height, width = cnn_features.size()
        cnn_features = cnn_features.view(batch, channels * height, width)
        cnn_features = cnn_features.permute(0, 2, 1)
        rnn_out, _ = self.rnn(cnn_features)
        output = self.fc(rnn_out)
        return output

def ctc_decoder(predictions, characters=CHARACTERS, blank_label=BLANK_LABEL):
    decoded = []
    for pred in predictions:
        chars = []
        prev_char = blank_label
        for p in pred:
            if p != blank_label and p != prev_char:
                chars.append(p.item())
                prev_char = p
            elif p == blank_label:
                prev_char = p
        decoded_str = ''.join([characters[c] for c in chars])
        decoded.append(decoded_str)
    return decoded

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = CaptchaCRNN().to(device)
criterion = nn.CTCLoss(blank=BLANK_LABEL, zero_infinity=True)
optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=3)

best_val_loss = float('inf')
patience = 5
trigger_times = 0

for epoch in range(EPOCHS):
    model.train()
    train_loss = 0.0
    correct = 0
    total = 0
    
    for images, labels, lengths in tqdm(train_loader, desc=f'Train {epoch+1}/{EPOCHS}'):
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        
        input_lengths = torch.full(size=(images.size(0),), 
                                  fill_value=outputs.size(1), 
                                  dtype=torch.long)
        loss = criterion(outputs.permute(1, 0, 2).log_softmax(2), 
                        labels, 
                        input_lengths, 
                        lengths)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
        
        preds = outputs.argmax(2).cpu().numpy()
        decoded_preds = ctc_decoder(preds)
        true_labels = [label[:length].tolist() for label, length in zip(labels.cpu(), lengths)]
        decoded_true = [''.join([CHARACTERS[c] for c in label]) for label in true_labels]
        
        correct += sum(p == t for p, t in zip(decoded_preds, decoded_true))
        total += len(decoded_preds)

    train_acc = correct / total
    
    model.eval()
    val_loss = 0.0
    correct_val = 0
    total_val = 0
    
    with torch.no_grad():
        for images, labels, lengths in tqdm(val_loader, desc=f'Val {epoch+1}/{EPOCHS}'):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            
            loss = criterion(outputs.permute(1, 0, 2).log_softmax(2), 
                            labels, 
                            torch.full((images.size(0),), outputs.size(1), dtype=torch.long), 
                            lengths)
            val_loss += loss.item()
            
            preds = outputs.argmax(2).cpu().numpy()
            decoded_preds = ctc_decoder(preds)
            true_labels = [label[:length].tolist() for label, length in zip(labels.cpu(), lengths)]
            decoded_true = [''.join([CHARACTERS[c] for c in label]) for label in true_labels]
            
            correct_val += sum(p == t for p, t in zip(decoded_preds, decoded_true))
            total_val += len(decoded_preds)
    
    val_acc = correct_val / total_val
    scheduler.step(val_loss)
    
    print(f'Epoch {epoch+1}/{EPOCHS} | '
          f'Train Loss: {train_loss/len(train_loader):.4f} Acc: {train_acc*100:.2f}% | '
          f'Val Loss: {val_loss/len(val_loader):.4f} Acc: {val_acc*100:.2f}%')
    
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save(model.state_dict(), 'captcha_model_best.pth')
        trigger_times = 0
    else:
        trigger_times += 1
        if trigger_times >= patience:
            print(f'Early stopping at epoch {epoch+1}')
            break

model.load_state_dict(torch.load('captcha_model_best.pth'))
model.eval()