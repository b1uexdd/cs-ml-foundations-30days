import torch
from loss import mae_loss, mse_loss, rmse_loss, huber_loss


def main():
    y_pred = torch.tensor([2.0, 4.0, 6.0, 8.0])
    y_target = torch.tensor([1.0, 5.0, 3.0, 10.0])

    print("y_pred:", y_pred)
    print("y_target:", y_target)
    print("-" * 40)

    # MAE
    my_mae = mae_loss(y_pred, y_target, reduction="mean")
    torch_mae = torch.nn.L1Loss(reduction="mean")(y_pred, y_target)

    print("My MAE:", my_mae.item())
    print("Torch MAE:", torch_mae.item())
    print("MAE same:", torch.allclose(my_mae, torch_mae))
    print("-" * 40)

    # MSE
    my_mse = mse_loss(y_pred, y_target, reduction="mean")
    torch_mse = torch.nn.MSELoss(reduction="mean")(y_pred, y_target)

    print("My MSE:", my_mse.item())
    print("Torch MSE:", torch_mse.item())
    print("MSE same:", torch.allclose(my_mse, torch_mse))
    print("-" * 40)

    # RMSE
    my_rmse = rmse_loss(y_pred, y_target, reduction="mean")
    expected_rmse = torch.sqrt(torch_mse + 1e-8)

    print("My RMSE:", my_rmse.item())
    print("Expected RMSE:", expected_rmse.item())
    print("RMSE same:", torch.allclose(my_rmse, expected_rmse))
    print("-" * 40)

    # Huber Loss
    my_huber = huber_loss(y_pred, y_target, threshold=1.0, reduction="mean")
    torch_huber = torch.nn.HuberLoss(delta=1.0, reduction="mean")(y_pred, y_target)

    print("My Huber Loss:", my_huber.item())
    print("Torch Huber Loss:", torch_huber.item())
    print("Huber same:", torch.allclose(my_huber, torch_huber))
    print("-" * 40)

    # reduction = none
    print("MAE reduction='none':", mae_loss(y_pred, y_target, reduction="none"))
    print("MSE reduction='none':", mse_loss(y_pred, y_target, reduction="none"))
    print("RMSE reduction='none':", rmse_loss(y_pred, y_target, reduction="none"))
    print("Huber reduction='none':", huber_loss(y_pred, y_target, threshold=1.0, reduction="none"))


if __name__ == "__main__":
    main()