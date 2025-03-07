import torch
from torchvision import transforms
from PIL import Image
import os
import string

CHARACTERS = string.ascii_lowercase + string.digits + string.ascii_uppercase
BLANK_LABEL = len(CHARACTERS)
IMG_WIDTH = 72
IMG_HEIGHT = 24

class CaptchaCRNN(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.cnn = torch.nn.Sequential(
            torch.nn.Conv2d(1, 64, 3, padding=1),
            torch.nn.BatchNorm2d(64),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2),
            torch.nn.Dropout2d(0.3),
            torch.nn.Conv2d(64, 128, 3, padding=1),
            torch.nn.BatchNorm2d(128),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2),
            torch.nn.Dropout2d(0.3),
            torch.nn.Conv2d(128, 256, 3, padding=1),
            torch.nn.BatchNorm2d(256),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2),
            torch.nn.Dropout2d(0.3)
        )
        self.rnn = torch.nn.GRU(256*(IMG_HEIGHT//8), 512, 3, 
                               bidirectional=True, batch_first=True, dropout=0.3)
        self.fc = torch.nn.Linear(1024, len(CHARACTERS)+1)

    def forward(self, x):
        x = self.cnn(x)
        x = x.view(x.size(0), -1, x.size(3))
        x = x.permute(0, 2, 1)
        x, _ = self.rnn(x)
        x = self.fc(x)
        return x

def preprocess_image(image_path):
    transform = transforms.Compose([
        transforms.Resize((IMG_HEIGHT, IMG_WIDTH)),
        transforms.Grayscale(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.6], std=[0.3])
    ])
    image = Image.open(image_path).convert('L')
    return transform(image).unsqueeze(0)

def ctc_decode(output):
    output = output.argmax(2).squeeze().cpu().numpy()
    decoded = []
    prev = -1
    for c in output:
        if c != prev and c != BLANK_LABEL:
            decoded.append(c)
        prev = c
    return ''.join(CHARACTERS[i] for i in decoded if i < len(CHARACTERS))

if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = CaptchaCRNN().to(device)
    model.load_state_dict(torch.load('captcha_model_best.pth', map_location=device))
    model.eval()

    captcha_dir = input("Введите путь к папке с CAPTCHA: ").strip()
    if not os.path.isdir(captcha_dir):
        raise ValueError("Указанный путь не существует или не является папкой")

    print("\nРаспознавание CAPTCHA:")
    for filename in os.listdir(captcha_dir):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            try:
                image_path = os.path.join(captcha_dir, filename)
                image_tensor = preprocess_image(image_path).to(device)
                
                with torch.no_grad():
                    output = model(image_tensor)
                
                captcha_text = ctc_decode(output)
                
                print(f"Файл: {filename:20} | Распознано: {captcha_text.upper()}")
                
            except Exception as e:
                print(f"Ошибка при обработке {filename}: {str(e)}")