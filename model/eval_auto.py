import torch


# =========================
# EVALUATION FUNCTION
# =========================
def evaluate(model, loader, y_mean, y_std, noise_std=0.0):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for x, y in loader:

            x = x.to(device)
            y = y.to(device)

            # Optional noise for robustness testing
            if noise_std > 0:
                x = x + torch.randn_like(x) * noise_std

            preds = model(x)

            all_preds.append(preds.cpu())
            all_labels.append(y.cpu())

    preds = torch.cat(all_preds)
    labels = torch.cat(all_labels)

    # =========================
    # DENORMALIZATION
    # =========================
    preds = preds * y_std + y_mean
    labels = labels * y_std + y_mean

    # =========================
    # METRICS
    # =========================
    mae = torch.mean(torch.abs(preds - labels), dim=0)
    rmse = torch.sqrt(torch.mean((preds - labels) ** 2, dim=0))

    ss_res = torch.sum((labels - preds) ** 2, dim=0)
    ss_tot = torch.sum((labels - torch.mean(labels, dim=0)) ** 2, dim=0)

    ss_tot = torch.where(ss_tot == 0, torch.ones_like(ss_tot), ss_tot)

    r2 = 1 - (ss_res / ss_tot)

    return {
        "mae": mae,
        "rmse": rmse,
        "r2": r2
    }