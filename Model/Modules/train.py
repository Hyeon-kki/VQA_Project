from tqdm import tqdm
import torch

# device
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
print(f"current device is {device}")

def train(model, loader, optimizer, criterion):
    model.train()
    total_loss = 0

    for data in tqdm(loader, total=len(loader)):
        # 컴퓨팅 자원을 할당하기 위해 사용한다.  
        images = data['image'].to(device) # torch.Size([64, 3, 224, 224])
        question = data['question'].to(device) # torch.Size([64, 32])
        answer = data['answer'].to(device)# torch.Size([64, 32])

        optimizer.zero_grad() # 배치당 그레디언트를 제거해버림 

        outputs = model(images, question) # torch.Size([64, 32, 50258])
    
        # Vocab size == 50258 토크나이저의 단어집
        # output: [batch, sequence, vocab], answer : [batch, sequence]
        # outputs.view(-1, outputs.size(-1))의 shape는 torch.Size([2048, 50258])
        # answer.view(-1))의 shape는 torch.Size(2048)

        loss = criterion(outputs.view(-1, outputs.size(-1)), answer.view(-1))
        total_loss += loss.item()

        loss.backward()
        optimizer.step()

    avg_loss = total_loss / len(loader)
    return avg_loss