import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import math

class GMMQuantile(torch.autograd.Function):
    """
    Custom Autograd Function for numerical root-finding of a Gaussian Mixture Model CDF
    and backpropagating exact analytic gradients via the Implicit Function Theorem.
    """
    @staticmethod
    def forward(ctx, mu, sigma, xi):
        # mu: (N, n2) - The conditional means for N paths and n2 morbidities
        # sigma: (n2,) - The additive noise scales
        N, n2 = mu.shape
        
        # Bisection search bounds spanning 5 standard deviations
        low = torch.min(mu, dim=0).values - 5.0 * sigma
        high = torch.max(mu, dim=0).values + 5.0 * sigma
        
        # Vectorized bisection search (~40 iterations saturates float32 precision)
        for _ in range(40):
            mid = (low + high) / 2.0
            
            # Evaluate GMM CDF at 'mid'
            z = (mid.unsqueeze(0) - mu) / sigma.unsqueeze(0)
            cdf_components = 0.5 * (1.0 + torch.erf(z / math.sqrt(2)))
            cdf_marg = torch.mean(cdf_components, dim=0) # Shape: (n2,)
            
            # Narrow bounds
            mask = cdf_marg < xi
            low = torch.where(mask, mid, low)
            high = torch.where(~mask, mid, high)
            
        q_star = (low + high) / 2.0
        
        # Save exact values needed for the IFT backward pass
        ctx.save_for_backward(mu, sigma, q_star)
        return q_star

    @staticmethod
    def backward(ctx, grad_q_star):
        mu, sigma, q_star = ctx.saved_tensors
        N = mu.shape[0]
        
        # Evaluate component PDFs at the root q_star
        z = (q_star.unsqueeze(0) - mu) / sigma.unsqueeze(0)
        pdf_components = torch.exp(-0.5 * z**2) / (math.sqrt(2 * math.pi) * sigma.unsqueeze(0))
        
        # Marginal PDF is the mean of component PDFs
        f_marg = torch.mean(pdf_components, dim=0) 
        
        # Implicit Function Theorem: dq*/dmu_i = (1/N * pdf_component_i) / f_marg
        # Adding 1e-12 prevents division by zero in the far tails
        dq_dmu = (pdf_components / N) / (f_marg.unsqueeze(0) + 1e-12)
        
        # Chain rule: multiply upstream gradient by local Jacobian
        grad_mu = grad_q_star.unsqueeze(0) * dq_dmu
        
        # Return gradients for inputs: (mu, sigma, xi) -> sigma and xi don't need gradients here
        return grad_mu, None, None


class SupplyOptimizer(nn.Module):
    def __init__(self, n2, num_samples, history_steps, delta, tau, lam=1.0, xi=0.95):
        super().__init__()
        self.n2 = n2
        self.N = num_samples
        self.history_steps = history_steps 
        self.delta = delta
        self.tau = tau
        
        self.lam = lam
        self.xi = xi
        
        self.s = nn.Parameter(torch.full((n2,), 20000.0))        
        self.rho = 10.0
        
        # conflict
        mean_conflict = 25
        self.theta_0 = 1. 
        self.theta_1 = 1 - self.theta_0/mean_conflict 
        self.sigma_E0 = .5
        
        # precipitation
        self.alpha_0 = 5.0
        self.alpha_1 = 0.3
        self.alpha_2 = -2.5
        self.alpha_3 = -4.0
        self.sigma_E1 = .5
        
        # population
        self.beta_0 =   0. 
        self.beta_1 =   1. 
        self.beta_2 =   0 
        self.beta_3 =   5 
        self.sigma_P0 = 0.5 
        
        # Gaussian morbidity parameters (Distinct for EVERY morbidity)
        self.marginal_incidence = 0.01 + 0.04 * torch.rand(n2) 
        self.marginal_incidence[0] = 0.011  
        
        self.beta_D = torch.rand(n2, 2) * 0.002 
        self.beta_D[0, 0] = 0.0             
        self.beta_D[0, 1] = 0.026           
        
        self.beta_D_next = torch.rand(n2, 2) * 0.002
        self.beta_D_next[0, 0] = 0.0        
        self.beta_D_next[0, 1] = 0.0        
        
        self.sigma_M = 10.0 + 50.0 * torch.rand(n2)  
        self.sigma_M[0] = 50.0              

    def soft_min(self, M, S):
        return S - F.softplus(S - M, beta=self.rho)

    def forward(self, D_t, P_t, M_t, S_t):
        D = D_t.repeat(self.N, 1)
        P = P_t.repeat(self.N)
        M = M_t.repeat(self.N, 1)
        S = S_t.repeat(self.N, 1)
        
        total_steps = self.history_steps + self.delta + self.tau
        
        # Container to track the conditional mean at the final step
        mu_gap_final = None 
        
        for k in range(total_steps):
            t = k + 1 
            
            if k == self.history_steps + self.delta:
                S = S + self.s
            
            T = self.soft_min(M, S)
            
            eps_D0 = torch.randn(self.N)
            eps_D1 = torch.randn(self.N)
            D_next_0 = self.theta_0 + self.theta_1 * D[:, 0] + self.sigma_E0 * eps_D0
            seasonality = self.alpha_2 * math.sin(2 * math.pi * t / 12) + self.alpha_3 * math.cos(2 * math.pi * t / 12)
            D_next_1 = F.relu(self.alpha_0 + self.alpha_1 * D[:, 1] + seasonality + self.sigma_E1 * eps_D1)
            D_next = torch.stack([D_next_0, D_next_1], dim=1)
            
            eps_P = torch.randn(self.N)
            P_next = (self.beta_0 + self.beta_1 * P + 
                      self.beta_2 * D[:, 0] + self.beta_3 * (D_next[:, 0]**2) + 
                      self.sigma_P0 * eps_P)
            
            U_prev = M - T 
            susceptible_pop = F.relu(P_next.unsqueeze(1) - U_prev)
            
            rate_adjustments = (torch.matmul(D, self.beta_D.T) + 
                                torch.matmul(D_next, self.beta_D_next.T))
            linear_rate = self.marginal_incidence + rate_adjustments
            
            # The exact conditional mean of the morbidities
            mu_M = U_prev + susceptible_pop * linear_rate
            
            eps_M = torch.randn(self.N, self.n2)
            new_cases = susceptible_pop * linear_rate + self.sigma_M * eps_M
            M_next = U_prev + new_cases

            S_next = S - T
            
            # At the final step, extract the conditional mean of the gap (M - S)
            if k == total_steps - 1:
                mu_gap_final = mu_M - S_next
            
            D, P, M, S = D_next, P_next, M_next, S_next
            
        # STEP 4 & 5: Exact Numerical Root-Finding with IFT Gradients
        # Pass the conditional means of the untreated gap into the autograd function
        q_gap = GMMQuantile.apply(mu_gap_final, self.sigma_M, self.xi)
        
        # The untreated cases are bounded below by zero (U = max(0, M - S))
        Q = F.relu(q_gap)
        
        cost = self.lam * torch.norm(self.s, p=2)**2
        penalty = torch.norm(Q, p=2)**2
        
        return cost + penalty

    @torch.no_grad()
    def get_expected_outcomes(self, D_t, P_t, M_t, S_t):
        D = D_t.repeat(self.N, 1)
        P = P_t.repeat(self.N)
        M = M_t.repeat(self.N, 1)
        S = S_t.repeat(self.N, 1)
        
        total_steps = self.history_steps + self.delta + self.tau
        
        for k in range(total_steps):
            t = k + 1 
            
            if k == self.history_steps + self.delta:
                S = S + self.s
            
            T = self.soft_min(M, S)
            
            eps_D0 = torch.randn(self.N)
            eps_D1 = torch.randn(self.N)
            D_next_0 = self.theta_0 + self.theta_1 * D[:, 0] + self.sigma_E0 * eps_D0
            seasonality = self.alpha_2 * math.sin(2 * math.pi * t / 12) + self.alpha_3 * math.cos(2 * math.pi * t / 12)
            D_next_1 = F.relu(self.alpha_0 + self.alpha_1 * D[:, 1] + seasonality + self.sigma_E1 * eps_D1)
            D_next = torch.stack([D_next_0, D_next_1], dim=1)
            
            eps_P = torch.randn(self.N)
            P_next = (self.beta_0 + self.beta_1 * P + 
                      self.beta_2 * D[:, 0] + self.beta_3 * (D_next[:, 0]**2) + 
                      self.sigma_P0 * eps_P)
            
            U_prev = M - T 
            susceptible_pop = F.relu(P_next.unsqueeze(1) - U_prev)
            
            rate_adjustments = (torch.matmul(D, self.beta_D.T) + 
                                torch.matmul(D_next, self.beta_D_next.T))
            linear_rate = self.marginal_incidence + rate_adjustments
            
            eps_M = torch.randn(self.N, self.n2)
            new_cases = susceptible_pop * linear_rate + self.sigma_M * eps_M
            M_next = U_prev + new_cases

            S_next = S - T
            D, P, M, S = D_next, P_next, M_next, S_next
            
        expected_M = torch.mean(M, dim=0)
        expected_S = torch.mean(S, dim=0)
        return expected_M, expected_S

# ============================================================================
# UPDATED PLOTTING FUNCTION (Shared Y-Limits & Fixed Title)
# ============================================================================
plt.rcParams['font.family'] = 'serif'

def plot_system_dynamics(histories, delta_line, plot_window=15, title=None, k=10, sim_lw=0.75, sim_ls='-', sim_color='gray', sim_alpha=0.4, y_limits=None):
    total_steps = histories['D'].shape[0]
    
    if plot_window == 'all' or plot_window >= total_steps:
        plot_window = total_steps
        
    months = np.arange(1, plot_window + 1)
    
    total_untreated_full = np.sum(np.maximum(0, histories['M'] - histories['S']), axis=2)
    
    variables = [
        ('Conflicts\n(Events)', histories['D'][-plot_window:, :, 0], 'tab:blue'),
        ('Population\n(Count)', histories['P'][-plot_window:, :], 'tab:blue'),
        ('Precip.\n(Inches)', histories['D'][-plot_window:, :, 1], 'tab:blue'),
        ('Malaria\n(Cases)', histories['M'][-plot_window:, :, 0], 'tab:blue'),
        ('Malaria Drug\n(Supply)', histories['S'][-plot_window:, :, 0], 'tab:blue'),
        ('Total Untreated\n(All Morbids)', total_untreated_full[-plot_window:, :], 'tab:red')
    ]
    
    num_plots = len(variables)
    fig, axes = plt.subplots(num_plots, 1, figsize=(10, 1.5 * num_plots))
    if title:
        fig.suptitle(title, fontsize=16)
        
    n_samples = histories['D'].shape[1]
    np.random.seed(42) 
    sim_indices = np.random.choice(n_samples, size=min(k, n_samples), replace=False)
    
    alpha_val = 0.25
    linewidth_val = 1.5
    linestyle_val = '-'
    
    for i, (ax, (ylabel, data, color)) in enumerate(zip(axes, variables)):
        median_val = np.median(data, axis=1)
        q25 = np.quantile(data, 0.25, axis=1)
        q75 = np.quantile(data, 0.75, axis=1)
        
        ax.plot(months, data[:, sim_indices], color=sim_color, 
                linewidth=sim_lw, linestyle=sim_ls, alpha=sim_alpha, zorder=1)
        ax.plot(months, median_val, color=color, linewidth=linewidth_val, linestyle=linestyle_val, zorder=3)
        ax.fill_between(months, q25, q75, color=color, alpha=alpha_val, zorder=2)
        
        ax.set_ylabel(ylabel, fontsize=12)
        ax.set_xticks(months)
        ax.grid(True, linestyle='--', alpha=0.5)
        
        if y_limits is not None:
            ax.set_ylim(y_limits[i])
        
        eval_window = 15 
        history_offset = max(0, plot_window - eval_window)
        adjusted_delta_line = history_offset + delta_line + 1
        
        ax.axvline(x=adjusted_delta_line, color='black', linestyle=':', alpha=0.7)
            
    axes[-1].set_xlabel('Months', fontsize=16, labelpad=10)
    fig.align_ylabels(axes)
    
    plt.tight_layout()
    if title:
        fig.subplots_adjust(top=0.92) 
    plt.show()


# ============================================================================
# UPDATED SIMULATOR FOR OPTIMIZED METHOD (27-Month Horizon, Gaussian)
# ============================================================================
@torch.no_grad()
def simulate_and_collect_histories(model, D_init, P_init, M_init, S_init, N=1000):
    history_steps = 12
    total_steps = history_steps + model.delta + model.tau 
    
    D = D_init.repeat(N, 1)
    P = P_init.repeat(N)
    M = M_init.repeat(N, 1)
    S = S_init.repeat(N, 1)
    
    histories = {
        'D': torch.zeros(total_steps, N, 2),
        'P': torch.zeros(total_steps, N),
        'M': torch.zeros(total_steps, N, model.n2),
        'S': torch.zeros(total_steps, N, model.n2),
    }
    
    for k in range(total_steps):
        t = k + 1
        
        if k == history_steps + model.delta:
            S = S + model.s
        
        T = model.soft_min(M, S)
        
        eps_D0 = torch.randn(N)
        eps_D1 = torch.randn(N)
        D_next_0 = model.theta_0 + model.theta_1 * D[:, 0] + model.sigma_E0 * eps_D0
        seasonality = model.alpha_2 * math.sin(2 * math.pi * t / 12) + model.alpha_3 * math.cos(2 * math.pi * t / 12)
        D_next_1 = F.relu(model.alpha_0 + model.alpha_1 * D[:, 1] + seasonality + model.sigma_E1 * eps_D1)
        D_next = torch.stack([D_next_0, D_next_1], dim=1)
        
        eps_P = torch.randn(N)
        P_next = (model.beta_0 + model.beta_1 * P + 
                  model.beta_2 * D[:, 0] + model.beta_3 * (D_next[:, 0]**2) + 
                  model.sigma_P0 * eps_P)
        
        # --- NEW GAUSSIAN MORBIDITY SAMPLING ---
        U_prev = M - T 
        susceptible_pop = F.relu(P_next.unsqueeze(1) - U_prev)
        
        rate_adjustments = (torch.matmul(D, model.beta_D.T) + 
                            torch.matmul(D_next, model.beta_D_next.T))
        linear_rate = model.marginal_incidence + rate_adjustments
        
        eps_M = torch.randn(N, model.n2)
        new_cases = susceptible_pop * linear_rate + model.sigma_M * eps_M
        M_next = U_prev + new_cases

        S_next = S - T
        
        histories['D'][k] = D_next
        histories['P'][k] = P_next
        histories['M'][k] = M_next
        histories['S'][k] = S_next
        
        D, P, M, S = D_next, P_next, M_next, S_next
        
    return {k: v.numpy() for k, v in histories.items()}


# ============================================================================
# UN BASELINE SIMULATOR (Gaussian)
# ============================================================================
@torch.no_grad()
def simulate_un_baseline(model, D_init, P_init, M_init, S_init, N=1000):
    history_steps = 12
    total_steps = history_steps + model.delta + model.tau 
    
    D_hist = torch.zeros(total_steps, N, 2)
    P_hist = torch.zeros(total_steps, N)
    
    D = D_init.repeat(N, 1)
    P = P_init.repeat(N)
    
    for k in range(total_steps):
        t = k + 1
        eps_D0 = torch.randn(N)
        eps_D1 = torch.randn(N)
        D_next_0 = model.theta_0 + model.theta_1 * D[:, 0] + model.sigma_E0 * eps_D0
        seasonality = model.alpha_2 * math.sin(2 * math.pi * t / 12) + model.alpha_3 * math.cos(2 * math.pi * t / 12)
        D_next_1 = F.relu(model.alpha_0 + model.alpha_1 * D[:, 1] + seasonality + model.sigma_E1 * eps_D1)
        D_next = torch.stack([D_next_0, D_next_1], dim=1)
        
        eps_P = torch.randn(N)
        P_next = (model.beta_0 + model.beta_1 * P + 
                  model.beta_2 * D[:, 0] + model.beta_3 * (D_next[:, 0]**2) + 
                  model.sigma_P0 * eps_P)
        
        D_hist[k] = D_next
        P_hist[k] = P_next
        D = D_next
        P = P_next

    M_hist = torch.zeros(total_steps, N, model.n2)
    S_hist = torch.zeros(total_steps, N, model.n2)
    T_hist = torch.zeros(total_steps, N, model.n2)
    
    M = M_init.repeat(N, 1)
    S = S_init.repeat(N, 1)
    
    for k in range(total_steps):
        if k == history_steps + model.delta:
            T_sum = torch.sum(T_hist[0:history_steps], dim=0)
            Unmet_final = F.relu(M_hist[history_steps-1] - S_hist[history_steps-1])
            Demand = T_sum + Unmet_final
            
            P_hist_window = P_hist[0:history_steps]
            P_hist_mean = torch.mean(P_hist_window, dim=0)
            
            t_hist = torch.arange(float(history_steps))
            X = torch.stack([torch.ones(history_steps), t_hist], dim=1)
            
            beta = torch.linalg.lstsq(X, P_hist_window).solution
            
            fut_start = history_steps + model.delta
            fut_end = total_steps
            t_fut_mean = torch.mean(torch.arange(float(fut_start), float(fut_end)))
            
            P_fut_extrapolated = beta[0] + beta[1] * t_fut_mean
            
            lam = torch.clamp(P_fut_extrapolated / P_hist_mean, min=0.0)
            
            Target_Stock = lam.unsqueeze(1) * Demand
            UN_order = F.relu(Target_Stock - S)
            
            S = S + UN_order
            
        T = model.soft_min(M, S)
        T_hist[k] = T
        
        D_curr = D_init.repeat(N, 1) if k == 0 else D_hist[k-1]
        P_next = P_hist[k]
        D_next = D_hist[k]
        
        # --- NEW GAUSSIAN MORBIDITY SAMPLING ---
        U_prev = M - T 
        susceptible_pop = F.relu(P_next.unsqueeze(1) - U_prev)
        
        rate_adjustments = (torch.matmul(D_curr, model.beta_D.T) + 
                            torch.matmul(D_next, model.beta_D_next.T))
        linear_rate = model.marginal_incidence + rate_adjustments
        
        eps_M = torch.randn(N, model.n2)
        new_cases = susceptible_pop * linear_rate + model.sigma_M * eps_M
        M_next = U_prev + new_cases
        
        S_next = S - T
        
        M_hist[k] = M_next
        S_hist[k] = S_next
        
        M = M_next
        S = S_next
        
    return {
        'D': D_hist.numpy(),
        'P': P_hist.numpy(),
        'M': M_hist.numpy(),
        'S': S_hist.numpy()
    }


# ============================================================================
# UPDATED PLOTTING FUNCTION (Side-by-Side Grid)
# ============================================================================
plt.rcParams['font.family'] = 'serif'

def plot_side_by_side(opt_histories, un_histories, delta_line, plot_window=15, title=None, k=10, sim_lw=0.75, sim_ls='-', sim_color='gray', sim_alpha=0.4, y_limits=None):
    total_steps = opt_histories['D'].shape[0]
    
    if plot_window == 'all' or plot_window >= total_steps:
        plot_window = total_steps
        
    months = np.arange(1, plot_window + 1)
    
    opt_untreated = np.sum(np.maximum(0, opt_histories['M'] - opt_histories['S']), axis=2)
    un_untreated = np.sum(np.maximum(0, un_histories['M'] - un_histories['S']), axis=2)
    
    def get_plot_vars(h, unt):
        return [
            ('Conflicts\n(Events)', h['D'][-plot_window:, :, 0], 'tab:blue'),
            ('Pop.\n(Count)', h['P'][-plot_window:, :], 'tab:blue'),
            ('Precip.\n(Inches)', h['D'][-plot_window:, :, 1], 'tab:blue'),
            ('Malaria\n(Cases)', h['M'][-plot_window:, :, 0], 'tab:blue'),
            ('Ma. Drug\nSupply', h['S'][-plot_window:, :, 0], 'tab:blue'),
            ('Untreated\n(All Morbids)', unt[-plot_window:, :], 'tab:red')
        ]
        
    vars_opt = get_plot_vars(opt_histories, opt_untreated)
    vars_un = get_plot_vars(un_histories, un_untreated)
    
    num_plots = len(vars_opt)
    fig, axes = plt.subplots(num_plots, 2, figsize=(16, 1.8 * num_plots))
    if title:
        fig.suptitle(title, fontsize=22)
        
    axes[0, 0].set_title("Policy optimization with quantile constraints", fontsize=20, pad=10)
    axes[0, 1].set_title("Previous year average adjustment", fontsize=20, pad=10)
        
    n_samples = opt_histories['D'].shape[1]
    np.random.seed(42) 
    sim_indices = np.random.choice(n_samples, size=min(k, n_samples), replace=False)
    
    alpha_val = 0.25
    linewidth_val = 1.5
    linestyle_val = '-'
    
    for col_idx, var_list in enumerate([vars_opt, vars_un]):
        for row_idx, (ylabel, data, color) in enumerate(var_list):
            ax = axes[row_idx, col_idx]
            
            median_val = np.median(data, axis=1)
            q25 = np.quantile(data, 0.25, axis=1)
            q75 = np.quantile(data, 0.75, axis=1)
            
            ax.plot(months, data[:, sim_indices], color=sim_color, 
                    linewidth=sim_lw, linestyle=sim_ls, alpha=sim_alpha, zorder=1)
            ax.plot(months, median_val, color=color, linewidth=linewidth_val, linestyle=linestyle_val, zorder=3)
            ax.fill_between(months, q25, q75, color=color, alpha=alpha_val, zorder=2)
            
            if col_idx == 0:
                ax.set_ylabel(ylabel, fontsize=16)
                
            ax.set_xticks(months)
            ax.grid(True, linestyle='--', alpha=0.5)
            
            if y_limits is not None:
                ax.set_ylim(y_limits[row_idx])
            
            eval_window = 15 
            history_offset = max(0, plot_window - eval_window)
            adjusted_delta_line = history_offset + delta_line + 1
            
            ax.axvline(x=adjusted_delta_line, color='black', linestyle=':', alpha=0.7)
            
    axes[-1, 0].set_xlabel('Months', fontsize=18, labelpad=10)
    axes[-1, 1].set_xlabel('Months', fontsize=18, labelpad=10)
    
    fig.align_ylabels(axes[:, 0])
    
    plt.tight_layout()
    if title:
        fig.subplots_adjust(top=0.92) 
    plt.show()


def plot_final_surplus_depletion_with_variance(opt_histories, un_histories, title="Surplus and Depletion at End of Procurement Period"):
    """
    Plots side-by-side bar charts showing surplus/depletion with an integrated 
    quantile density envelope showing the variance across the N Monte Carlo paths.
    """
    # 1. Extract final time step data: Shape (N, n2)
    opt_M_final = opt_histories['M'][-1]
    opt_S_final = opt_histories['S'][-1]
    
    un_M_final = un_histories['M'][-1]
    un_S_final = un_histories['S'][-1]
    
    # 2. Calculate the exact Surplus/Depletion metric per sample path
    # Shape remains (N, n2)
    opt_metric_paths = opt_S_final - opt_M_final
    un_metric_paths = un_S_final - un_M_final
    
    # 3. Compute expected values (means) for the bars
    opt_mean = np.mean(opt_metric_paths, axis=0)
    un_mean = np.mean(un_metric_paths, axis=0)
    
    # 4. Compute density percentiles for the variance envelopes
    opt_p10, opt_p25, opt_p75, opt_p90 = np.percentile(opt_metric_paths, [10, 25, 75, 90], axis=0)
    un_p10, un_p25, un_p75, un_p90 = np.percentile(un_metric_paths, [10, 25, 75, 90], axis=0)
    
    n2 = opt_mean.shape[0]
    indices = np.arange(n2)
    
    fig, (ax2, ax1) = plt.subplots(1, 2, figsize=(16, 6), sharey=True)
    #fig.suptitle(title, fontsize=18)#, y=1.02)
    
    # --- UN Approach Plot (Left) ---
    un_colors = ['tab:blue' if val >= 0 else 'tab:red' for val in un_mean]
    ax1.bar(indices, un_mean, color=un_colors, width=1.0, edgecolor='black', linewidth=0.3, zorder=2)
    
    # Overlay discrete density envelopes using step='mid' to match bar widths
    ax1.fill_between(indices, un_p25, un_p75, color='black', alpha=0.3, step='mid', zorder=3, label='IQR (25th-75th)')
    ax1.fill_between(indices, un_p10, un_p90, color='black', alpha=0.15, step='mid', zorder=3, label='10th-90th Percentile')
    
    ax1.set_title("Previous year average adjustment", fontsize=15, pad=12)
    ax1.set_xlabel("Morbidity Index", fontsize=14)
    ax1.set_ylabel("Depletion | Surplus", fontsize=13)
    ax1.axhline(0, color='black', linewidth=1.5, zorder=4)
    ax1.grid(True, axis='y', linestyle='--', alpha=0.7, zorder=1)
    ax1.legend(loc='upper right')
    
    # --- Dynamic Optimization Plot (Right) ---
    opt_colors = ['tab:blue' if val >= 0 else 'tab:red' for val in opt_mean]
    ax2.bar(indices, opt_mean, color=opt_colors, width=1.0, edgecolor='black', linewidth=0.3, zorder=2)
    
    ax2.fill_between(indices, opt_p25, opt_p75, color='black', alpha=0.3, step='mid', zorder=3)
    ax2.fill_between(indices, opt_p10, opt_p90, color='black', alpha=0.15, step='mid', zorder=3)
    
    ax2.set_title("Policy optimization with quantile constraints", fontsize=15, pad=12)
    ax2.set_xlabel("Morbidity Index", fontsize=14)
    ax2.axhline(0, color='black', linewidth=1.5, zorder=4)
    ax2.grid(True, axis='y', linestyle='--', alpha=0.7, zorder=1)
    
    # Format the y-axis with commas for readability
    ax1.get_yaxis().set_major_formatter(plt.FuncFormatter(lambda x, loc: "{:,}".format(int(x))))
    
    # Center the zero line symmetrically
    y_min, y_max = ax1.get_ylim()
    bound = max(abs(y_min), abs(y_max))
    #ax1.set_ylim(-bound, bound)
    ax1.set_ylim(y_min, y_max)
    
    plt.tight_layout()
    plt.show()

# ============================================================================
# Optimization
# ============================================================================

# --- Simulation Setup ---
n2_morbidities = 100
num_mc_samples = 10
history_months = 12
delta_months = 3
tau_months = 12

# Lagrangian Parameters
lambda_penalty = 1e-1
target_quantile = 0.95

D_init = torch.tensor([0.0, 2.0])
P_init = torch.tensor(10000.0)
M_init = torch.tensor([1000.0 for j in range(n2_morbidities)])
S_init = torch.tensor([30000 for j in range(n2_morbidities)])

# Pass history_months and lagrangian parameters to the model
model = SupplyOptimizer(n2_morbidities, num_mc_samples, history_months, delta_months, tau_months, lam=lambda_penalty, xi=target_quantile)
optimizer = torch.optim.Adam(model.parameters(), lr=20.0)

loss_history = []
num_epochs = 10000

for epoch in range(num_epochs):
    optimizer.zero_grad()
    loss = model(D_init, P_init, M_init, S_init)
    loss.backward()
    optimizer.step()
    
    with torch.no_grad():
        model.s.clamp_(min=0.0)
        
    loss_history.append(loss.item())




# ============================================================================
# Simulation and Plotting
# ============================================================================
num_simulation_samples = 10000
plot_win = 15 

print("Simulating models...")
optimized_histories = simulate_and_collect_histories(
    model, D_init, P_init, M_init, S_init, N=num_simulation_samples
)
un_histories = simulate_un_baseline(
    model, D_init, P_init, M_init, S_init, N=num_simulation_samples
)

opt_untreated = np.sum(np.maximum(0, optimized_histories['M'] - optimized_histories['S']), axis=2)
un_untreated = np.sum(np.maximum(0, un_histories['M'] - un_histories['S']), axis=2)

bw = 5
def get_global_limits(data1, data2, lower_pct=bw, upper_pct=100-bw):
    min_val = min(np.percentile(data1, lower_pct), np.percentile(data2, lower_pct))
    max_val = max(np.percentile(data1, upper_pct), np.percentile(data2, upper_pct))
    margin = (max_val - min_val) * 0.05 if max_val > min_val else 1.0
    lower_bound = min_val - margin
    if min_val >= 0 and lower_bound < 0:
        lower_bound = 0.0
    return (lower_bound, max_val + margin)

shared_y_limits = [
    get_global_limits(optimized_histories['D'][-plot_win:, :, 0], un_histories['D'][-plot_win:, :, 0]),
    get_global_limits(optimized_histories['P'][-plot_win:, :], un_histories['P'][-plot_win:, :]),
    get_global_limits(optimized_histories['D'][-plot_win:, :, 1], un_histories['D'][-plot_win:, :, 1]),
    get_global_limits(optimized_histories['M'][-plot_win:, :, 0], un_histories['M'][-plot_win:, :, 0]),
    get_global_limits(optimized_histories['S'][-plot_win:, :, 0], un_histories['S'][-plot_win:, :, 0]),
    get_global_limits(opt_untreated[-plot_win:, :], un_untreated[-plot_win:, :])
]

print("Rendering comparison plot...")
plot_side_by_side(
    optimized_histories, 
    un_histories,
    delta_line=delta_months, 
    plot_window=plot_win,
    y_limits=shared_y_limits,
    k=15, sim_lw=1.0, sim_ls='-', sim_color='black', sim_alpha=0.1
)

plot_final_surplus_depletion_with_variance(optimized_histories, un_histories, title="Surplus and depletion at end of procurement period")


