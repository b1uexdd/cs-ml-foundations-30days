import torch
import torch.nn as nn

logits = torch.tensor([
    [2.0, 1.0, 0.1],
    [0.5, 2.5, 0.3]
])

targets = torch.tensor([0, 1])

#loss = - correct_class_logit + log(sum(exp(all_logits)))

batch_indices = torch.arange(logits.shape[0]) #【0，1】
target_logits = logits[batch_indices, targets] #【2，2.5】

exp_logits = torch.exp(logits) 

sum_exp = exp_logits.sum(dim=1)

log_sum_exp = torch.log(sum_exp)

loss_each = -target_logits + log_sum_exp

loss_manual = loss_each.mean()

loss_torch = nn.CrossEntropyLoss()(logits, targets)

print("target logits:", target_logits)
print("manual loss:", loss_manual)
print("torch loss:", loss_torch)
print("same:", torch.allclose(loss_manual, loss_torch))
    