import os
import pandas as pd
from PIL import Image
from tqdm.auto import tqdm
from torch.utils.data import Dataset
import warnings 
warnings.filterwarnings('ignore')

class VQADataset(Dataset): # 아룰 상속함으로써 Dataloder와 상호작용 가능하다. 
    def __init__(self, df, tokenizer, transform, img_path, is_test=False):
        self.df = df
        self.tokenizer = tokenizer
        self.transform = transform
        self.img_path = img_path
        self.is_test = is_test
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        # Image 경로를 구성한다.
        img_name = os.path.join(self.img_path, row['image_id'] + '.jpg')
        # Image file을 열고 색공간을 RGB로 변환 
        image = Image.open(img_name).convert('RGB')
        image = self.transform(image)

        question = row['question']
        question = self.tokenizer.encode_plus(
            question,
            truncation = True,
            add_special_tokens=True,
            max_length = 32,
            padding='max_length',
            return_attention_mask = True,
            return_tensors='pt'
        )

        if not self.is_test:
            answer = row['answer']
            answer = self.tokenizer.encode_plus(
                answer,
                max_length = 32,
                padding = 'max_length',
                truncation = True,
                return_tensors='pt'
            )
            return { 
                'image': image.squeeze(), # 1인 차원 제거 
                'question': question['input_ids'].squeeze(),
                'answer': answer['input_ids'].squeeze() # 'input_ids'는 토크나이저 통과해서 나온 토큰 별 id
            }
        else:
            return {
                'image': image,
                'question': question['input_ids'].squeeze(),
            }
