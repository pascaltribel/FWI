import numpy as np
import matplotlib.pyplot as plt
import torch
from tqdm.auto import tqdm

def damping(nx, ny, size, alpha=1.):
    damping = torch.ones((nx, ny))
    x = torch.arange(nx).repeat(ny, 1).T
    y = torch.arange(ny).repeat(nx, 1)
    dist_to_edge = torch.minimum(
        torch.minimum(x, nx - 1 - x),
        torch.minimum(y, ny - 1 - y)
    )
    damping_factor = torch.exp(-alpha * dist_to_edge)
    damping *= damping_factor
    return damping.T

def gradient(f, dim, spacing, order=4):
    if isinstance(dim, int):
        dim = (dim,)
    if isinstance(spacing, (float, int)):
        spacing = (spacing,) * len(dim)
        
    coefficients = {
        1: (torch.tensor([-1, 1]), [-1, 1]),
        2: (torch.tensor([-1/2, 1/2]), [-1, 1]),
        4: (torch.tensor([-1/12, 8/12, -8/12, 1/12]), [-2, -1, 1, 2]),
        6: (torch.tensor([1/60, -3/20, 3/4, -3/4, 3/20, -1/60]), [-3, -2, -1, 1, 2, 3]),
        10: (torch.tensor([-1/252, 1/21, -1/4, 4/3, -3, 3, -4/3, 1/4, -1/21, 1/252]), 
              [-5, -4, -3, -2, -1, 1, 2, 3, 4, 5])
    }
    
    if order not in coefficients:
        raise ValueError("Ordre doit être 1, 2, 4, 6 ou 10")
        
    coeffs, shifts = coefficients[order]
    
    gradients = []
    for d, h in zip(dim, spacing):
        grad = torch.zeros_like(f)
        for coeff, shift in zip(coeffs, shifts):
            grad += coeff * torch.roll(f, shifts=shift, dims=d)
        grad /= h
        gradients.append(grad)
        
    return tuple(gradients)

def step(v_xp, v_xs, v_zp, v_zs, u, w, alpha, beta, dx, dt, nx, ny, damping_size):
    damping_tensor = damping(nx, ny, damping_size)
    du_dx, du_dz = gradient(u, dim=(0, 1), spacing=dx)
    dw_dx, dw_dz = gradient(w, dim=(0, 1), spacing=dx)
    A = du_dx + dw_dz
    B = du_dz - dw_dx
    dA_dx, dA_dz = gradient(A, dim=(0, 1), spacing=dx)
    dB_dx, dB_dz = gradient(B, dim=(0, 1), spacing=dx)
    dv_xp_dt = alpha**2 * dA_dx
    dv_xs_dt = beta**2 * dB_dz
    dv_zp_dt = alpha**2 * dA_dz
    dv_zs_dt = -beta**2 * dB_dx
    v_xp += dt * dv_xp_dt
    v_xs += dt * dv_xs_dt
    v_zp += dt * dv_zp_dt
    v_zs += dt * dv_zs_dt
    v_x = v_xp + v_xs
    v_z = v_zp + v_zs
    du_dt = v_x
    dw_dt = v_z
    u += dt * du_dt
    w += dt * dw_dt
    v_xp, v_xs, v_zp, v_zs = v_xp*(1-damping_tensor), v_xs*(1-damping_tensor), v_zp*(1-damping_tensor), v_zs*(1-damping_tensor)
    return v_xp, v_xs, v_zp, v_zs, u, w

def simulate(v_xp, v_xs, v_zp, v_zs, u, w, alpha, beta, dx, nx, ny, dt, nt, damping_size):
    trajectory = [torch.concatenate((u.unsqueeze(-1), w.unsqueeze(-1)), axis=-1).unsqueeze(0)]
    for _ in tqdm(range(nt)):
        v_xp, v_xs, v_zp, v_zs, u, w = step(v_xp, v_xs, v_zp, v_zs, u, w, alpha, beta, dx, dt, nx, ny, damping_size)
        trajectory.append(torch.concatenate((u.unsqueeze(-1), w.unsqueeze(-1)), axis=-1).unsqueeze(0))
    return torch.vstack(trajectory)

def initialise_conditions(nx, ny, mu, sigma=5.):
    v_xp, v_xs, v_zp, v_zs = torch.zeros((ny, nx)), torch.zeros((ny, nx)), torch.zeros((ny, nx)), torch.zeros((ny, nx))
    u, w = torch.zeros((ny, nx)), torch.zeros((ny, nx))
    sigma = 5.0
    x = torch.arange(0, nx).float()
    y = torch.arange(0, ny).float()
    X, Y = torch.meshgrid(x, y, indexing="xy")
    dist_squared = (X - mu[0])**2 + (Y - mu[1])**2
    f = torch.exp(-dist_squared / (2 * sigma**2))
    df_dx = -(X - mu[0]) * f / sigma**2
    df_dy = -(Y - mu[1]) * f / sigma**2
    norm = torch.sqrt(df_dx**2 + df_dy**2).max()
    g_x = df_dx / norm
    g_y = df_dy / norm
    theta = torch.tensor([torch.pi / 4])
    u = g_x * torch.cos(theta) - g_y * torch.sin(theta)
    w = g_x * torch.sin(theta) + g_y * torch.cos(theta)
    return v_xp, v_xs, v_zp, v_zs, u, w

def add_random_shapes(matrix, base_value, num_shapes=3, damping_size=16):
    ny, nx = matrix.shape
    result = torch.full((ny, nx), base_value)

    for _ in range(num_shapes):
        h, w = torch.randint(nx//10, nx//5, (2,))
        x, y = torch.randint(damping_size, nx-damping_size, (2,))
        shape_value = base_value * (1+torch.normal(0, 0.25, (1,)))
        result[x:x+h, y:y+w] = shape_value
    return result

def plot_trajectory(trajectory, t):
    fig, ax = plt.subplots(1, 2, figsize=(15, 5))
    ax[0].imshow(torch.log(torch.sqrt(trajectory[:, 0, :, 0]**2+trajectory[:, 0, :, 1]**2)), aspect='auto')
    ax[0].set_title("Trace at the surface")
    ax[0].set_xlabel("Position")
    ax[0].set_ylabel("Time")
    ax[1].imshow(torch.sqrt(trajectory[t, :, :, 0]**2+trajectory[t, :, :, 1]**2), aspect='auto')
    ax[1].set_title("Final displacement field")
    ax[1].set_xlabel("Horizontal position")
    ax[1].set_xlabel("Vertical position")
    plt.tight_layout()
    plt.show()