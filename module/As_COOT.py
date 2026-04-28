import torch
import torch.nn as nn
import torch.nn.functional as F


class AsCOOT(nn.Module):
    def __init__(self, epsilon=0.01, rho_x=5.0, max_iter_outer=5, max_iter_inner=100):

        super().__init__()
        self.epsilon = epsilon
        self.rho_x = rho_x
        self.max_iter_outer = max_iter_outer
        self.max_iter_inner = max_iter_inner

    def get_coot_cost(self, X, Y, pi_dual):

        m1 = torch.sum(pi_dual, dim=1, keepdim=True)
        m2 = torch.sum(pi_dual, dim=0, keepdim=True)

        if pi_dual.shape[0] == X.shape[1] and pi_dual.shape[1] == Y.shape[1]:
            r_pi_f = torch.sum(pi_dual, dim=1).view(-1, 1)
            c_pi_f = torch.sum(pi_dual, dim=0).view(1, -1)
            t1 = (X ** 2) @ r_pi_f
            t2 = (Y ** 2) @ c_pi_f.T
            const_term = t1 + t2.T 
            cross_term = -2 * (X @ pi_dual @ Y.T)
        else:
            r_pi_s = torch.sum(pi_dual, dim=1).view(-1, 1)
            c_pi_s = torch.sum(pi_dual, dim=0).view(1, -1)
            t1 = (X.T ** 2) @ r_pi_s
            t2 = (Y.T ** 2) @ c_pi_s.T
            const_term = t1 + t2.T
            cross_term = -2 * (X.T @ pi_dual @ Y)

        return const_term + cross_term

    def s_uot_sinkhorn(self, C, mu, nu, rho, is_balanced_nu=True):
        device = C.device

        c_min = C.min()
        c_max = C.max()
        if c_max > c_min + 1e-8:
            C_norm = (C - c_min) / (c_max - c_min)
        else:
            C_norm = torch.zeros_like(C)
        #print(C_norm)

        K = torch.exp(-C_norm / self.epsilon)

        b = torch.ones_like(nu).to(device)
        a = torch.ones_like(mu).to(device)

        if rho == float('inf'):
            gamma = 1.0
        else:
            gamma = rho / (rho + self.epsilon)

        for _ in range(self.max_iter_inner):
            Kb = K @ b
            a = (mu / (Kb + 1e-16)) ** gamma

            KTa = K.T @ a
            if is_balanced_nu:
                b = nu / (KTa + 1e-16)
            else:
                b = (nu / (KTa + 1e-16)) ** gamma

        R = torch.diag(a.flatten()) @ K @ torch.diag(b.flatten())
        return R

    def forward(self, X, Y):

        device = X.device
        N, d1 = X.shape
        M, d2 = Y.shape

        mu_s = (torch.ones(N) / N).to(device)
        nu_s = (torch.ones(M) / M).to(device)
        mu_f = (torch.ones(d1) / d1).to(device)
        nu_f = (torch.ones(d2) / d2).to(device)

        pi_s = mu_s.unsqueeze(1) @ nu_s.unsqueeze(0)
        pi_f = mu_f.unsqueeze(1) @ nu_f.unsqueeze(0)

        for i in range(self.max_iter_outer):
            C_s = self.get_coot_cost(X, Y, pi_f)

            if i == 0:
                print(f"Iter 0 Raw Cost Range: {C_s.max().item() - C_s.min().item():.6f}")

            pi_s_new = self.s_uot_sinkhorn(C_s, mu_s, nu_s, rho=self.rho_x, is_balanced_nu=True)
            mass_ratio = pi_f.sum() / (pi_s_new.sum() + 1e-16)
            pi_s = pi_s_new * mass_ratio

            C_f = self.get_coot_cost(X, Y, pi_s)
            pi_f_new = self.s_uot_sinkhorn(C_f, mu_f, nu_f, rho=float('inf'), is_balanced_nu=True)
            mass_ratio_f = pi_s.sum() / (pi_f_new.sum() + 1e-16)
            pi_f = pi_f_new * mass_ratio_f

        return pi_s, pi_f


