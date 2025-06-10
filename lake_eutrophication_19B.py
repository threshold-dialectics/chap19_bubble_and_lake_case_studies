# lake_eutrophication_19B.py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
import seaborn as sns

# Set a professional plotting theme
sns.set_theme(style="whitegrid", context="talk")

class LakeEcosystem:
    """
    A class to simulate the dynamics of lake eutrophication, representing a slow-burn
    regime shift through the lens of Threshold Dialectics.
    """
    def __init__(self, n_years=40, random_seed=2024):
        """
        Initializes the simulation parameters and the lake's starting state.
        
        Args:
            n_years (int): Total number of years to simulate.
            random_seed (int): Seed for reproducibility.
        """
        self.n_years = n_years
        np.random.seed(random_seed)

        # Core State Variables
        self.nutrient_loading = 0.1  # strain_p proxy (starts low)
        self.dissolved_oxygen = 0.9  # fcrit_p proxy (starts high)
        self.sav_dominance = 0.8     # beta_p proxy (starts high, stable)
        self.indicator_biodiversity = 0.85 # g_p proxy
        
        # Tipping point threshold
        self.oxygen_collapse_threshold = 0.25
        self.has_collapsed = False
        
        # Data history list
        self.history = []

    def _record_state(self, year):
        """Records the current state of all variables and proxies."""
        state = {
            'year': year,
            'nutrient_loading': self.nutrient_loading,
            'dissolved_oxygen': self.dissolved_oxygen,
            'sav_dominance': self.sav_dominance,
            'indicator_biodiversity': self.indicator_biodiversity,
            'beta_p': self.sav_dominance,
            'fcrit_p': self.dissolved_oxygen,
            'g_p': self.indicator_biodiversity,
            'strain_p': self.nutrient_loading,
            'has_collapsed': self.has_collapsed
        }
        self.history.append(state)
        
    def _update_step(self, year):
        """Contains the core dynamic logic for a single simulation step (year)."""
        # 1. External pressure slowly and relentlessly increases
        # This represents increased agricultural/urban runoff over decades
        self.nutrient_loading += 0.015 + np.random.normal(0, 0.005)

        if not self.has_collapsed:
            # --- Pre-Collapse Dynamics ---
            # 2. Oxygen (Slack) is consumed by decomposing nutrients
            oxygen_drain = self.nutrient_loading * 0.08
            self.dissolved_oxygen -= oxygen_drain
            
            # 3. SAV dominance (Rigidity) is highly resilient... at first.
            # It changes very little, showing the system's apparent stability.
            self.sav_dominance -= np.random.normal(0.005, 0.002)

            # 4. Check for the tipping point
            if self.dissolved_oxygen < self.oxygen_collapse_threshold:
                self.has_collapsed = True
                print(f"--- Regime Shift Triggered at Year {year}! ---")
                # Rapid collapse of the SAV-dominated state
                self.sav_dominance = 0.1 + np.random.uniform(-0.05, 0.05)
        else:
            # --- Post-Collapse Dynamics ---
            # The system is now in a stable, algae-dominated state.
            self.sav_dominance = 0.1 + np.random.normal(0, 0.02)
            # Oxygen remains low because there's no SAV to produce it
            self.dissolved_oxygen = 0.15 + np.random.normal(0, 0.02)
            
        # Universal dynamics
        # Indicator species biodiversity depends on system health (oxygen)
        self.indicator_biodiversity = self.dissolved_oxygen * 0.9

        # Clip all values to stay within reasonable bounds
        self.nutrient_loading = np.clip(self.nutrient_loading, 0, 1.0)
        self.dissolved_oxygen = np.clip(self.dissolved_oxygen, 0, 1.0)
        self.sav_dominance = np.clip(self.sav_dominance, 0, 1.0)
        self.indicator_biodiversity = np.clip(self.indicator_biodiversity, 0, 1.0)

    def run(self):
        """Executes the full simulation."""
        for year in range(self.n_years):
            self._update_step(year)
            self._record_state(year)
            
        return pd.DataFrame(self.history).set_index('year')

def calculate_td_diagnostics(df, couple_window=5, smooth_window=5):
    """Calculates TD diagnostics. Uses a shorter window due to lower data frequency."""
    if smooth_window % 2 == 0: smooth_window += 1
    
    df['beta_dot'] = savgol_filter(df['beta_p'], window_length=smooth_window, polyorder=2, deriv=1)
    df['fcrit_dot'] = savgol_filter(df['fcrit_p'], window_length=smooth_window, polyorder=2, deriv=1)
    
    df['SpeedIndex'] = np.sqrt(df['beta_dot']**2 + df['fcrit_dot']**2)
    df['CoupleIndex'] = df['beta_dot'].rolling(window=couple_window).corr(df['fcrit_dot'])
    
    # Scale speed for better visualization relative to other examples
    df['SpeedIndex'] *= 5 
    
    return df.dropna()

def plot_eutrophication_cascade(df):
    """Generates a 2x2 plot for the full eutrophication lifecycle."""
    fig, axes = plt.subplots(2, 2, figsize=(20, 16))
    collapse_year = df[df['has_collapsed']].index.min()
    fig.suptitle('Threshold Dialectics of a Lake Eutrophication Regime Shift', fontsize=28, y=0.98)
    
    # --- Panel 1: Lever Proxies ---
    ax = axes[0, 0]
    ax.plot(df.index, df['beta_p'], color='blue', lw=3, label=r'$\beta_p$ (SAV Dominance)')
    ax.plot(df.index, df['fcrit_p'], color='red', lw=3, label=r'$F_{crit,p}$ (Dissolved Oxygen)')
    ax.axvline(collapse_year, color='black', linestyle='--', lw=2.5, label='Regime Shift')
    ax.set_title('Lever Proxies vs. Time', fontsize=18)
    ax.set_ylabel('Normalized Value', fontsize=16)
    ax.set_ylim(-0.05, 1.05)
    ax.legend(loc='best', fontsize=14)
    
    # --- Panel 2: Systemic Strain ---
    ax = axes[0, 1]
    ax.plot(df.index, df['strain_p'], color='black', lw=3, label=r'$\langle\Delta_P\rangle_p$ (Nutrient Loading)')
    ax.axvline(collapse_year, color='black', linestyle='--', lw=2.5)
    ax.set_title('Systemic Strain Proxy vs. Time', fontsize=18)
    ax.set_ylabel('Strain Level', fontsize=16)
    ax.set_ylim(-0.05, 1.05)
    ax.legend(loc='best', fontsize=14)
         
    # --- Panel 3: TD Diagnostics ---
    ax = axes[1, 0]
    ax.plot(df.index, df['SpeedIndex'], color='darkorange', lw=3, label='Speed Index')
    ax.plot(df.index, df['CoupleIndex'], color='purple', lw=3, label='Couple Index')
    ax.axhline(0, color='gray', linestyle=':', lw=2)
    ax.axvline(collapse_year, color='black', linestyle='--', lw=2.5)
    ax.set_title('TD Diagnostics vs. Time', fontsize=18)
    ax.set_ylabel('Index Value', fontsize=16)
    ax.set_ylim(-1.05, 1.05)
    ax.legend(loc='best', fontsize=14)
    
    # --- Panel 4: S/C Diagnostic Plane ---
    ax = axes[1, 1]
    pre_collapse_df = df[df.index < collapse_year]
    post_collapse_df = df[df.index >= collapse_year]

    # Risk zones
    ax.axvspan(0.1, 0.2, color='yellow', alpha=0.3)
    ax.axvspan(0.2, 0.5, color='red', alpha=0.3)
    
    # Plot pre-collapse trajectory
    ax.plot(pre_collapse_df['SpeedIndex'], pre_collapse_df['CoupleIndex'], color='green', lw=2.5, label='Pre-Shift Trajectory')
    ax.scatter(pre_collapse_df['SpeedIndex'].iloc[0], pre_collapse_df['CoupleIndex'].iloc[0], color='blue', s=200, zorder=5, label='Start (Healthy)')
    
    # Plot post-collapse state
    ax.scatter(post_collapse_df['SpeedIndex'], post_collapse_df['CoupleIndex'], color='brown', s=100, zorder=4, alpha=0.6, label='Post-Shift State')
    ax.scatter(pre_collapse_df['SpeedIndex'].iloc[-1], pre_collapse_df['CoupleIndex'].iloc[-1], color='red', marker='X', s=250, zorder=5, label='Tipping Point')

    ax.set_title('Trajectory on Diagnostic Plane', fontsize=18)
    ax.set_xlabel('Speed Index', fontsize=16)
    ax.set_ylabel('Couple Index', fontsize=16)
    ax.set_xlim(0, 0.4)
    ax.set_ylim(-1.05, 1.05)
    ax.legend(loc='best', fontsize=14)

    for ax_row in axes:
        for ax_col in ax_row:
            if ax_col is not axes[1, 1]:
                ax_col.set_xlabel('Year', fontsize=16)
            ax_col.tick_params(axis='both', which='major', labelsize=14)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig("lake_eutrophication_dashboard.png", dpi=300, bbox_inches='tight')
    plt.show()

# --- Main Execution ---
if __name__ == "__main__":
    # 1. Instantiate and run the simulation
    lake_sim = LakeEcosystem(n_years=40, random_seed=2024)
    history_df = lake_sim.run()

    # 2. Calculate TD diagnostics
    full_df = calculate_td_diagnostics(history_df, couple_window=5, smooth_window=5)

    # 3. Plot the full lifecycle
    plot_eutrophication_cascade(full_df)