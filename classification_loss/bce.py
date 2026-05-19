"""
今日任务：手写 BCE、BCE with logits

logits → softmax → CE
logit → sigmoid → BCE

BCE(y,p)=-[ylog(p)+(1-y)log(1-p)]
BCEWithLogits


BCEWithLogits(x,y) = max(x,0) - xy + log(1+exp(-|x|))
"""

import torch
import torch.nn as nn

probs = torch.tensor([0.8, 0.2, 0.6])
targets = torch.tensor([1.0, 0.0, 1.0])

loss = - (torch.log(probs) * targets + torch.log(1 - probs) * (1 - targets))
#print(loss)

logits = torch.tensor([1.5, -1.0, 0.2])
targets = torch.tensor([1.0, 0.0, 1.0])

probs = torch.sigmoid(logits)
#print(probs)
loss = -(torch.log(probs) * targets + (1 - targets) * torch.log(1 - probs))
loss_manual = loss.mean()
#print(loss_manual)

loss_torch = nn.BCEWithLogitsLoss()(logits, targets)
print("manual BCE with logits:", loss_manual)
print("torch BCEWithLogitsLoss:", loss_torch)
print("same:", torch.allclose(loss_manual, loss_torch))