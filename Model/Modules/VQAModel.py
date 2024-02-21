import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import GPT2Tokenizer, GPT2Model # 텍스트
import torchvision.models as models # 이미지
from torchvision import transforms

class VQAModel(nn.Module):
    def __init__(self, vocab_size):
        super(VQAModel, self).__init__()
        self.vocab_size = vocab_size # 50258

        # ResNet-50의 출력 차원은 보통 1000입니다. 하지만, fine tunning을 통해 변환가능하다. 
        # 예를 들어, 전이 학습(Transfer Learning)을 사용하여 새로운 작업을 수행할 때는 출력 차원을 원하는 클래스 수에 맞게 조정할 수 있습니다.
        self.resnet = models.resnet50(pretrained=True) # 이미지 representation을 뽑아내는 resnet50이다.  

        self.gpt2 = GPT2Model.from_pretrained('gpt2') # 텍스트 representation을 뽑아내는 GPT2이다. 
        self.gpt2.resize_token_embeddings(vocab_size) # 기존의 vocab size에 추가한 [PAD] 토큰 반영한 것을 GPT2에 등록한다. 

        combined_features_size = 1000 + self.gpt2.config.hidden_size # resnet 출력 차원 + gpt2 출력 차원 (단순 Fusion)
        self.classifier = nn.Linear(combined_features_size, vocab_size)

    def forward(self, images, question):
        image_features = self.resnet(images) # torch.Size([64, 1000])
     
        image_features = image_features.view(image_features.size(0),-1) # 2차원으로 변환한다. (batch, 나머지) # torch.Size([64, 1000])
        outputs = self.gpt2(question)
        text_features = outputs.last_hidden_state # [batch, sequence, hidden] # torch.Size [64, 32, 768]

        # unsqueeze를 통해 dim == 1을 확장한다. 이후에, expand를 통해서 텐서의 차원을 변경하는데 -1은 유지 즉 dim == 0과 dim == 2는 유지하고 dim == 은 text_features.size(1)로 변환한다.  
        image_features = image_features.unsqueeze(1).expand(-1, text_features.size(1),-1) # [batch, sequence, 1000] # torch.Size [64, 32, 1000]
        combined = torch.cat([image_features, text_features], dim=-1) # [batch, sequence, 1000+hidden]  # torch.Size [64, 32, 1768]
        output = self.classifier(combined) # [batch, vocab_size]
        return output