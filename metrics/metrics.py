#1. multi-class accuracy
#2. binary accuracy
#3. top-k accuracy
#4. confusion matrix
#5. precision
#6. recall
#7. F1 score
#8. per-class accuracy
import torch

def accuracy(logits, targets):
    """
    Multi-class accuracy.

    logits: shape [batch_size, num_classes]
    targets: shape [batch_size]
    """
    preds = torch.argmax(logits, dim=1)
    correct = (preds == targets).float() #将true false变为1 0
    return correct.mean()

def binary_accuracy(y_pred, y_target, threshold=0.5, from_logits=True):
    """
    Binary classification accuracy.

    y_pred: shape [batch_size]
    y_target: shape [batch_size]
    """
    if from_logits:
        y_pred = torch.sigmoid(y_pred)
    else:
        y_pred = y_pred
    
    preds = y_pred >= threshold
    correct = (preds == y_target).float()

    return correct.mean()

def top_k_accuracy(logits, targets, k=5):
    """
    Top-k accuracy.

    The prediction is correct if the true label appears in the top-k predicted classes.
    """
    _, indices = torch.topk(logits, k, dim=-1)
    targets = targets.view(-1,1)
    correct = (indices == targets).float()
    return correct.mean()

def confusion_matrix_binary(y_pred, y_target, threshold=0.5, from_logits=True):
    """
    Binary confusion matrix.

    Returns:
        TP, TN, FP, FN
    """

    if from_logits:
        preds = torch.sigmoid(y_pred)
    else:
        preds = y_pred

    preds = preds >= threshold
    tp = ((preds == 1) & (y_target == 1)).sum()
    tn = ((preds == 0) & (y_target == 0)).sum()
    fp = ((preds == 1) & (y_target == 0)).sum()
    fn = ((preds == 0) & (y_target == 1)).sum()

    return tp, tn, fp, fn

def precision_binary(y_pred, y_target, threshold=0.5, from_logits=True):
    tp, tn, fp, fn = confusion_matrix_binary(
        y_pred,
        y_target,
        threshold=threshold,
        from_logits=from_logits
    )

    precision = tp / (tp + fp)
    return precision

def recall_binary(y_pred, y_target, threshold=0.5, from_logits=True):
    tp, tn, fp, fn = confusion_matrix_binary(
        y_pred,
        y_target,
        threshold=threshold,
        from_logits=from_logits
    )

    recall = tp / (tp + fn)  
    return recall

def f1_score_binary(y_pred, y_target, threshold=0.5, from_logits=True):
    precision = precision_binary(
        y_pred,
        y_target,
        threshold=threshold,
        from_logits=from_logits,
    )

    recall = recall_binary(
        y_pred,
        y_target,
        threshold=threshold,
        from_logits=from_logits,
    )

    f1 = 2 * precision * recall / (precision + recall)
    return f1

def per_class_accuracy(logits, targets, num_classes):
    """
    Compute accuracy for each class.

    Returns:
        Tensor with shape [num_classes]
    """
    preds = torch.argmax(logits, dim=1)

    acc_list = []

    for cls in range(num_classes):
        mask = targets ==cls
        if mask.sum() == 0:
            acc_list.append(torch.tensor(float("nan")))
        else:
            correct = (preds[mask] == targets[mask]).float().mean()
            acc_list.append(correct)

    return torch.stack(acc_list)
