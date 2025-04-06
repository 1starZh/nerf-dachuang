import torch

def plane_fit_loss(points):
    B, N, _ = points.shape
    centroid = points.mean(dim=1)
    centered = points - centroid[:, None, :]
    cov = torch.matmul(centered.transpose(1,2), centered) / N
    eigvals = torch.linalg.eigvalsh(cov)
    loss = eigvals[:, 0]  # 最小特征值代表偏离平面的程度
    return loss.mean()

def line_fit_loss(points):
    B, N, _ = points.shape
    centroid = points.mean(dim=1)
    centered = points - centroid[:, None, :]
    cov = torch.matmul(centered.transpose(1,2), centered) / N
    eigvals = torch.linalg.eigvalsh(cov)
    loss = eigvals[:, 1:].sum(dim=1)  # 排除最大主轴
    return loss.mean()
