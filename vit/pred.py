import numpy as np
from PIL import Image
import torch 
from model import VisionTransformer

k = 10

imagenet_labels = dict(enumerate(open("classes.txt")))

model = torch.load("model.pth")
model.eval()


img = (np.array(Image.open("cat.png"))/128) - 1  #in range of -1 and 1
inp = torch.from_numpy(img).permute(2,0,1).unsqueeze(0).to(torch.float32) 
logits = model(inp)
probs = torch.nn.functional.softmax(logits, dim=-1)

top_probs, top_classes = probs[0].topk(k)
print("Top 10 predictions are: \n" + "--"*29)
for i, (ix_, prob_) in enumerate(zip(top_classes,top_probs)):
    ix = ix_.item()
    prob = prob_.item()
    cls = imagenet_labels[ix].strip()

    print(f"{i}: {cls:<45} --- {prob:.4f}")