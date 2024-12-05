import torch

"""Given a matrix n x k where n < k, return n zero-mean orthonormal vectors in k-dimensional space."""

def bjorck(weights, beta=0.5, iters=15):
    H = weights - torch.mean(weights, dim=1, keepdim=True)
    H = H / (torch.linalg.matrix_norm(H, 2))
    for _ in range(iters):
        H_t_H = H.T @ H
        H = (1 + beta) * H - beta * H @ H_t_H
    return H