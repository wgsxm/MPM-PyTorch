import time
import torch
from tqdm import tqdm

from mpm_pytorch.constitutive_models.warp_svd import SVD

wp_svd = SVD()
wp_svd(torch.randn(1, 3, 3))
def warp_svd(x):
    return wp_svd(x)
    
def torch_svd(x):
    U, s, Vh = torch.svd(x)
    return U, s, Vh.transpose(-2, -1)

if __name__ == '__main__':
    test_time = 10
    n = 100000
    atol = 1e-5

    print("\nTest correctness")
    Is = torch.eye(3).unsqueeze(0).repeat(n, 1, 1)
    for _ in tqdm(range(test_time), desc="Test orthogonality of U and V"):
        x = torch.randn(n, 3, 3)
        U, s, V = warp_svd(x)
        UU = U @ U.transpose(-2, -1)
        VV = V @ V.transpose(-2, -1)
        assert torch.allclose(UU, Is, atol=1e-5)
        assert torch.allclose(VV, Is, atol=1e-5)
    bar = tqdm(range(test_time), desc="Test correctness of decomposition")
    for _ in bar:
        x = torch.randn(n, 3, 3)
        U, s, V = warp_svd(x)
        warp_pred = U @ torch.diag_embed(s) @ V
        warp_mae = torch.abs(warp_pred - x).mean()
        U, s, V = torch_svd(x)
        torch_pred = U @ torch.diag_embed(s) @ V
        torch_mae = torch.abs(torch_pred - x).mean()
        bar.set_postfix(warp_mae=warp_mae.item(), torch_mae=torch_mae.item())
        assert warp_mae < atol
    
    print("\nTest differentiability")
    bar = tqdm(range(test_time), desc="Test differentiability")
    for _ in bar:
        x = torch.randn(n, 3, 3, requires_grad=True)
        U, s, V = warp_svd(x)
        warp_pred = U @ torch.diag_embed(s) @ V
        warp_pred.sum().backward()
        assert x.grad is not None
        warp_mae = torch.abs(x.grad - torch.ones_like(x)).mean()
        x.grad = None
        U, s, V = torch_svd(x)
        torch_pred = U @ torch.diag_embed(s) @ V
        torch_pred.sum().backward()
        torch_mae = torch.abs(x.grad - torch.ones_like(x)).mean()
        bar.set_postfix(warp_mae=warp_mae.item(), torch_mae=torch_mae.item())
        assert warp_mae < atol
    
    print("\nTest speed")
    x = torch.randn(n, 3, 3)
    start = time.time()
    for _ in tqdm(range(test_time), desc="Test warp_svd forward (CPU)"):
        U, s, V = warp_svd(x)
    end = time.time()
    print("warp_svd: ", (end - start) / test_time, "seconds")
    start = time.time()
    for _ in tqdm(range(test_time), desc="Test torch_svd forward (CPU)"):
        U, s, V = torch_svd(x)
    end = time.time()
    print("torch_svd: ", (end - start) / test_time, "seconds")

    print()
    x = torch.randn(n, 3, 3, requires_grad=True)
    start = time.time()
    for _ in tqdm(range(test_time), desc="Test warp_svd forward + backward (CPU)"):
        U, s, V = warp_svd(x)
        warp_pred = U @ torch.diag_embed(s) @ V
        warp_pred.sum().backward()
    end = time.time()
    print("warp_svd: ", (end - start) / test_time, "seconds")
    x.grad = None
    start = time.time()
    for _ in tqdm(range(test_time), desc="Test torch_svd forward + backward (CPU)"):
        U, s, V = torch_svd(x)
        torch_pred = U @ torch.diag_embed(s) @ V
        torch_pred.sum().backward()
    end = time.time()
    print("torch_svd: ", (end - start) / test_time, "seconds")

    if torch.cuda.is_available():
        print()
        x = torch.randn(n, 3, 3).cuda().requires_grad_()
        start = time.time()
        for _ in tqdm(range(test_time), desc="Test warp_svd forward + backward (GPU)"):
            U, s, V = warp_svd(x)
            warp_pred = U @ torch.diag_embed(s) @ V
            warp_pred.sum().backward()
        end = time.time()
        print("warp_svd: ", (end - start) / test_time, "seconds")
        x.grad = None
        start = time.time()
        for _ in tqdm(range(test_time), desc="Test torch_svd forward + backward (GPU)"):
            U, s, V = torch_svd(x)
            torch_pred = U @ torch.diag_embed(s) @ V
            torch_pred.sum().backward()
        end = time.time()
        print("torch_svd: ", (end - start) / test_time, "seconds")

    
    
            