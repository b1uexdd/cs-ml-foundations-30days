import torch

from metrics import (
    accuracy,
    binary_accuracy,
    top_k_accuracy,
    confusion_matrix_binary,
    precision_binary,
    recall_binary,
    f1_score_binary,
    per_class_accuracy
)


def test_multiclass_metrics():
    logits = torch.tensor([
        [0.1, 0.8, 0.1],
        [2.0, 1.0, 0.5],
        [0.2, 0.3, 0.9],
        [0.7, 1.2, 0.1],
        [0.1, 0.2, 2.0],
    ])

    targets = torch.tensor([1, 0, 2, 0, 1])

    acc = accuracy(logits, targets)
    top2_acc = top_k_accuracy(logits, targets, k=2)
    class_acc = per_class_accuracy(logits, targets, num_classes=3)

    print("===== Multi-class Metrics =====")
    print("Logits:")
    print(logits)
    print("Targets:", targets)
    print("Accuracy:", acc.item())
    print("Top-2 Accuracy:", top2_acc.item())
    print("Per-class Accuracy:", class_acc)
    print()


def test_binary_metrics():
    binary_logits = torch.tensor([2.0, -1.0, 0.3, -0.8, 1.2, -2.0])
    binary_targets = torch.tensor([1, 0, 1, 0, 0, 1])

    acc = binary_accuracy(binary_logits, binary_targets, from_logits=True)
    tp, tn, fp, fn = confusion_matrix_binary(binary_logits, binary_targets, from_logits=True)

    precision = precision_binary(binary_logits, binary_targets, from_logits=True)
    recall = recall_binary(binary_logits, binary_targets, from_logits=True)
    f1 = f1_score_binary(binary_logits, binary_targets, from_logits=True)

    print("===== Binary Metrics =====")
    print("Binary logits:", binary_logits)
    print("Binary targets:", binary_targets)
    print("Binary Accuracy:", acc.item())
    print("TP:", tp.item())
    print("TN:", tn.item())
    print("FP:", fp.item())
    print("FN:", fn.item())
    print("Precision:", precision.item())
    print("Recall:", recall.item())
    print("F1 Score:", f1.item())
    print()


def main():
    test_multiclass_metrics()
    test_binary_metrics()


if __name__ == "__main__":
    main()