import os
import pandas as pd

import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

import torchvision.models as models # 이미지
from torchvision import transforms
from PIL import Image

from transformers import GPT2Tokenizer, GPT2Model # 텍스트

from tqdm.auto import tqdm
import warnings 
warnings.filterwarnings('ignore')

from QA_project.Model.Modules.train import train
from Modules.inference import inference
from Modules.VQADataset import VQADataset
from Modules.VQAModel import VQAModel
from Modules.load_data import load_data

# device
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
print(f"current device is {device}")

train_df, test_df, sample_submission, train_img_path, test_img_path = load_data()

# dataset & dataloader
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
tokenizer.add_special_tokens({'pad_token': '[PAD]'})
vocab_size = len(tokenizer)

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

train_dataset = VQADataset(train_df, tokenizer, transform, train_img_path, is_test=False)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)


# Model
model = VQAModel(vocab_size).to(device)

# Criterion and Optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), lr=5e-5)

# Training loop
for epoch in range(1):
    avg_loss = train(model, train_loader, optimizer, criterion)
    print(f"Epoch: {epoch+1}, Loss: {avg_loss:.4f}")