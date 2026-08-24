import numpy as np
from scipy.special import expit
import matplotlib.pyplot as plt

# 1. Set lead time, number of simulations, and constraints
m = 12
n_samples = 10000
S_1 = 30000.0  
constraint_C1 = 1000000.0 

# 2. Set reasonable model parameters 
# params = {
#     'mu_E0': 2.0, 'sigma_E0': 0.5,    
#     'mu_E1': 5.0, 'sigma_E1': 1.0,    
#     'theta_0': 0.5, 'theta_1': 0.7,   
#     'alpha_0': 1.0, 'alpha_1': 0.6, 'alpha_2': 2.0, 'alpha_3': 1.0, 
#     'mu_P0_0': 500.0, 'mu_P0_1': 50.0, 'sigma_P0': 20.0, 
#     'beta_0': 100.0, 'beta_1': 0.8, 'beta_2': 10.0, 'beta_3': 20.0, 
#     'gamma_0': -3.0, 'gamma_1': 0.4   
# }

params = {
    # Conflict Events (ACLED estimates for active regional conflict)
    'mu_E0': 25.0, 'sigma_E0': 8.0,    
    'theta_0': 5.0, 'theta_1': 0.8,   
    
    # Precipitation (Kiremt rainy season, measured in inches)
    'mu_E1': 2.0, 'sigma_E1': 1.5,    
    'alpha_0': 3.0, 'alpha_1': 0.3, 'alpha_2': -2.5, 'alpha_3': -4.0, 
    
    # IDP Population (Scaled to a moderate-to-large camp size)
    'mu_P0_0': 15000.0, 'mu_P0_1': 100.0, 'sigma_P0': 500.0, 
    'beta_0': 1000.0, 'beta_1': 0.95, 'beta_2': 20.0, 'beta_3': 50.0, 
    
    # Malaria Prevalence (Highland epidemic-prone dynamics)
    'gamma_0': -4.5, 'gamma_1': 0.35   
}

def sample_system_history(m, n_samples, params, S_1):
    """
    Simulates the system and tracks the full history of states for m months.
    Returns dictionaries of history arrays, each of shape (m, n_samples).
    """
    eps = np.random.normal(0, 1, (m + 1, n_samples))
    #
    # Initial conditions
    E_t_0 = params['mu_E0'] + params['sigma_E0'] * eps[0]
    E_t_1 = params['mu_E1'] + params['sigma_E1'] * eps[0]
    P_t_0 = params['mu_P0_0'] + params['mu_P0_1'] * E_t_0 + params['sigma_P0'] * eps[0]
    #
    S_current = np.full(n_samples, float(S_1))
    U_prev = np.zeros(n_samples)
    #
    # Initialize history arrays
    histories = {
        'conflict': np.zeros((m, n_samples)),
        'precipitation': np.zeros((m, n_samples)),
        'population': np.zeros((m, n_samples)),
        'malaria_cases': np.zeros((m, n_samples)),
        'drug_supply': np.zeros((m, n_samples))
    }
    #
    for t in range(1, m + 1):
        t_idx = t - 1
        E_prev_0, E_prev_1, P_prev_0 = E_t_0, E_t_1, P_t_0
        
        # System updates
        E_t_0 = params['theta_0'] + params['theta_1'] * E_prev_0 + params['sigma_E0'] * eps[t]
        E_t_1 = (params['alpha_0'] + params['alpha_1'] * E_prev_1 + 
                 params['alpha_2'] * np.sin(2 * np.pi * t / 12) + 
                 params['alpha_3'] * np.cos(2 * np.pi * t / 12) + params['sigma_E1'] * eps[t])
        
        P_t_0 = (params['beta_0'] + params['beta_1'] * P_prev_0 + 
                 params['beta_2'] * E_prev_0 + params['beta_3'] * E_t_0 + params['sigma_P0'] * eps[t])
        
        # Malaria cases using mechanistic carryover
        new_incidence = expit(params['gamma_0'] + params['gamma_1'] * E_prev_1)
        susceptible_pop = np.maximum(0, P_t_0 - U_prev)
        P_t_cases = U_prev + (susceptible_pop * new_incidence)
        
        # Record histories (Supply here is what is available AT the start of month t)
        histories['conflict'][t_idx] = E_t_0
        histories['precipitation'][t_idx] = E_t_1
        histories['population'][t_idx] = P_t_0
        histories['malaria_cases'][t_idx] = P_t_cases
        histories['drug_supply'][t_idx] = S_current.copy()
        
        # Calculate depletions and untreatred carryovers for the NEXT month
        prescriptions = np.minimum(S_current, P_t_cases)
        S_current = S_current - prescriptions
        U_prev = P_t_cases - prescriptions
    #
    return histories


def optimize_deliveries_over_time(malaria_history, supply_history, C1):
    """
    Computes optimal drug deliveries across the entire simulation history matrix.
    Both input arrays are of shape (m, n_samples).
    """
    # Target intervention capped by operational constraints
    S_optimal = np.minimum(malaria_history, C1)
    
    # We order the difference between the optimal target and what is currently in stock
    deliveries = np.maximum(0, S_optimal - supply_history)
    return deliveries

plt.rcParams['font.family'] = 'serif'
def plot_system_dynamics(histories, deliveries, m):
    """
    Plots the median and the (0.25, 0.75) quantiles for all requested variables.
    """
    months = np.arange(1, m + 1)
    variables = [
        ('Number\nconflicts', histories['conflict'], 'tab:red'),
        ('Camp\nPopulation', histories['population'], 'tab:purple'),
        ('Precip.\n(inches)', histories['precipitation'], 'tab:blue'),
        ('Malaria\nCases', histories['malaria_cases'], 'tab:orange'),
        ('Drug\nSupply', histories['drug_supply'], 'tab:green'),
        ('Optimal\nDelivery', deliveries, 'tab:brown')
    ]
    
    # --- FIXED: Calculate the month supplies fall short based on delivery need ---
    median_delivery = np.median(deliveries, axis=1)
    shortage_indices = np.where(median_delivery > 0)[0]
    
    if len(shortage_indices) > 0:
        run_out_month = months[shortage_indices[0]]
    else:
        run_out_month = None
    # -------------------------------------------------
    
    # Create a 6x1 vertical layout with an adjusted figure height
    fig, axes = plt.subplots(6, 1, figsize=(10, 8))
    ct = 1
    #
    alpha_val = 0.1
    alpha_val2 = 0.25
    linewidth_val = 1.25
    linewidth_val2 = 2
    linestyle_val = '-.'
    for ax, (ylabel, data, color) in zip(axes, variables):
        median_val = np.median(data, axis=1)
        q25 = np.quantile(data, 0.25, axis=1)
        q75 = np.quantile(data, 0.75, axis=1)
        
        if ct == 6 and run_out_month is not None:
            # Find the array index corresponding to the run_out_month
            idx = np.where(months == run_out_month)[0][0]
            
            # Plot the segment before and up to the run out point in blue
            ax.plot(months[:idx+1], median_val[:idx+1], color='blue', linewidth=linewidth_val,linestyle=linestyle_val)
            ax.fill_between(months[:idx+1], q25[:idx+1], q75[:idx+1], color='blue', alpha=alpha_val)
            
            # Plot the segment from the run out point onward in red
            ax.plot(months[idx:], median_val[idx:], color='red', linewidth=linewidth_val2,linestyle=linestyle_val)
            ax.fill_between(months[idx:], q25[idx:], q75[idx:], color='red', alpha=alpha_val2)
            
            # Add the vertical red dotted line
            #ax.axvline(x=run_out_month, color='red', linestyle=':', linewidth=linewidth_val)
        else:
            # Plot all other subplots normally in blue
            ax.plot(months, median_val, color='blue', linewidth=linewidth_val,linestyle=linestyle_val)
            ax.fill_between(months, q25, q75, color='blue', alpha=alpha_val)
        
        # Set the vertical y-axis label instead of a title
        ax.set_ylabel(ylabel, fontsize=15)
        
        ax.set_xticks(months)
        ax.grid(True, linestyle='-', alpha=0.6)
        ct += 1
            
    # Apply the x-axis label only to the last subplot in the stack
    axes[-1].set_xlabel('Month', fontsize=16, labelpad=10)
            
    # Align the y-labels nicely across all subplots
    fig.align_ylabels(axes)
    
    plt.tight_layout()
    plt.savefig('/Users/dcunha/Documents/FDM/system_dynamics.pdf', dpi=300, bbox_inches='tight')
    #plt.show()

# 3. Execution
histories = sample_system_history(m, n_samples, params, S_1)
optimal_deliveries = optimize_deliveries_over_time(
    histories['malaria_cases'], 
    histories['drug_supply'], 
    constraint_C1
)

# Render the multi-panel plot
plot_system_dynamics(histories, optimal_deliveries, m)