# device
from tqdm import tqdm
import torch
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
print(f"current device is {device}")

def inference(model, loader):
    model.eval()
    preds = []
    with torch.no_grad():
        for data in tqdm(loader, total=len(loader)):
            images = data['image'].to(device)
            question = data['question'].to(device)

            outputs = model(images, question) # [batch, sequence, vocab]

            _, pred = torch.max(outputs, dim=2) # values, indices = _, pred
            # .cpu() 메서드는 GPU에 있는 텐서를 CPU로 이동시켜 NumPy 변환이 가능하다.
            # .extend() 메서드는 리스트에 다른 리스트나 이터러블 객체의 모든 요소를 추가한다.
            preds.extend(pred.cpu().numpy()) 

    return preds