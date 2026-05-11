#| 模块           | 可以补充的内容                                      |
#| ------------ | -------------------------------------------- |
#| 基础 Loss      | 手写 `MAE`、`MSE`、`RMSE`                        |
#| reduction 机制 | 支持 `reduction="none" / "mean" / "sum"`       |
#| 梯度理解         | 推导 MSE、MAE、RMSE 对预测值的梯度                      |
#| PyTorch 对照   | 和 `torch.nn.MSELoss`、`torch.nn.L1Loss` 对比结果  |
#| 数值稳定性        | RMSE 里加 `eps`，避免除 0                          |
#| 扩展 Loss      | 补一个 `Huber Loss / Smooth L1 Loss`            |
#| 加权 Loss      | 手写 `Weighted MSE / Weighted MAE`             |
#| 单元测试         | 用 `assert torch.allclose()` 检查自己写的 loss 是否正确 |
#| 输出文件         | `losses.py` + `test_losses.py` + 简短 README   |

import torch

#mae
def mae_loss(y_pred, y_target, reduction="mean"):
    loss = torch.abs(y_pred - y_target)
    if reduction == "mean":
        return loss.mean()
    elif reduction == "sum":
        return loss.sum()
    else:
        return loss

#mse
def mse_loss(y_pred, y_target, reduction="mean"):
    loss = (y_pred - y_target) ** 2
    if reduction == "mean":
        return loss.mean()
    if reduction == "sum":
        return loss.sum()
    else:
        return loss

    
#rmse
def rmse_loss(y_pred, y_target, reduction="mean"):
    loss = mse_loss(y_pred, y_target, reduction=reduction)
    return torch.sqrt(loss)

#huber_loss
def huber_loss(y_pred, y_target, threshold=0.1, reduction="mean"):
    error = torch.abs(y_pred - y_target)
    loss = torch.where(error <= threshold, 0.5 * error ** 2, threshold * (error - 0.5 * threshold))
    if reduction == "mean":
        return loss.mean()
    if reduction == "sum":
        return loss.sum()
    else:
        return loss