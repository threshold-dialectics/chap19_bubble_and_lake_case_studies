# echo_bubble_19A.py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
import seaborn as sns
import scipy.stats as stats
# Set a professional plotting theme
sns.set_theme(style="whitegrid", context="talk")

class FinancialMarketSystem:
    """
    A class to simulate the dynamics of a financial market "Echo Bubble"
    through the lens of Threshold Dialectics.
    """
    def __init__(self, n_weeks=70, random_seed=101):
        """
        Initializes the simulation parameters and the market's starting state.
        
        Args:
            n_weeks (int): Total number of weeks to simulate.
            random_seed (int): Seed for the random number generator for reproducibility.
        """
        self.n_weeks = n_weeks
        self.phase_transitions = {
            'P1_Lull_End': 20,          # Changed from 10
            'P2_Emerge_End': 30,        # Shifted
            'P3_Inflate_End': 55,       # Shifted
            'P4_Fragility_End': 62,     # Shifted
            'P5_Correct_End': 66,       # Shifted
        }
        np.random.seed(random_seed)

        # Core State Variables
        self.price_deviation = 0.05  # strain_p proxy
        self.valuation_consensus = 0.4 # beta_p proxy (starts moderate)
        self.margin_debt = 0.2       # fcrit_p proxy (starts low)
        self.sentiment_velocity = 0.1 # g_p proxy (starts low)
        
        # Data history list
        self.history = []

    def _get_current_phase(self, week):
        """Determines the current phase based on the week number."""
        if week <= self.phase_transitions['P1_Lull_End']: return 'P1_Lull'
        if week <= self.phase_transitions['P2_Emerge_End']: return 'P2_Emerge'
        if week <= self.phase_transitions['P3_Inflate_End']: return 'P3_Inflate'
        if week <= self.phase_transitions['P4_Fragility_End']: return 'P4_Fragility'
        if week <= self.phase_transitions['P5_Correct_End']: return 'P5_Correct'
        return 'P6_Reset'

    def _record_state(self, week, phase):
        """Records the current state of all variables and proxies."""
        state = {
            'week': week,
            'phase': phase,
            'price_deviation': self.price_deviation,
            'valuation_consensus': self.valuation_consensus,
            'margin_debt': self.margin_debt,
            'sentiment_velocity': self.sentiment_velocity,
            'beta_p': self.valuation_consensus,
            'fcrit_p': self.margin_debt,
            'g_p': self.sentiment_velocity,
            'strain_p': self.price_deviation,
        }
        self.history.append(state)


    def _update_step(self, week, phase):
        """Contains the core dynamic logic for a single simulation step (week)."""
        # Original noise: noise = np.random.normal(0, 0.01)

        if phase == 'P1_Lull':
            # MAKE THE "QUIET" PHASE QUIETER
            noise = np.random.normal(0, 0.002) # <-- REDUCED NOISE
            self.price_deviation += noise
            self.margin_debt += noise * 0.5
        
        elif phase == 'P2_Emerge':
            noise = np.random.normal(0, 0.01)
            self.sentiment_velocity += 0.03 + noise
            self.price_deviation += 0.01 + self.sentiment_velocity * 0.1
            self.margin_debt += 0.01 + noise
        
        elif phase == 'P3_Inflate':
            # MAKE THE "LOUD" PHASE LOUDER
            noise = np.random.normal(0, 0.01)
            # Higher deviation justifies more debt
            self.margin_debt += (self.price_deviation * 0.15 + noise * 2) # <-- INCREASED COEFFICIENT from 0.1
            # Higher debt & sentiment reinforces consensus
            self.valuation_consensus += (self.margin_debt * 0.06 + self.sentiment_velocity * 0.02 + noise) # <-- INCREASED from 0.05
            # Higher consensus justifies higher deviation
            self.price_deviation += (self.valuation_consensus * 0.1 + noise * 1.5) # <-- INCREASED from 0.08
            self.sentiment_velocity += 0.01 + noise
            
        elif phase == 'P4_Fragility':
            noise = np.random.normal(0, 0.01)
            self.price_deviation += np.random.normal(0, 0.03)
            self.margin_debt += np.random.normal(0, 0.02)
            self.valuation_consensus += np.random.normal(0, 0.02)
            self.sentiment_velocity -= 0.05 # Sentiment starts to cool
            
        elif phase == 'P5_Correct':
            noise = np.random.normal(0, 0.01)
            self.price_deviation -= 0.25 + noise
            self.valuation_consensus -= 0.15 + noise
            self.margin_debt -= 0.20 + noise
            self.sentiment_velocity = 0.1 # Sentiment collapses
            
        elif phase == 'P6_Reset':
            noise = np.random.normal(0, 0.01)
            self.price_deviation += np.random.normal(-0.01, 0.01)
            self.margin_debt += np.random.normal(-0.01, 0.01)
            self.valuation_consensus += np.random.normal(0.005, 0.01)

        # Clip all values to stay within reasonable bounds
        self.price_deviation = np.clip(self.price_deviation, 0, 1.0)
        self.valuation_consensus = np.clip(self.valuation_consensus, 0.1, 1.0)
        self.margin_debt = np.clip(self.margin_debt, 0.1, 1.0)
        self.sentiment_velocity = np.clip(self.sentiment_velocity, 0.1, 1.0)

    def run(self):
        """Executes the full simulation."""
        for week in range(self.n_weeks):
            phase = self._get_current_phase(week)
            self._update_step(week, phase)
            self._record_state(week, phase)
            
        return pd.DataFrame(self.history).set_index('week')

def calculate_td_diagnostics(df, couple_window=10, smooth_window=7):
    """Calculates TD diagnostics (Speed, Couple) from the simulation history."""
    if smooth_window % 2 == 0: smooth_window += 1
    
    df['beta_dot'] = savgol_filter(df['beta_p'], window_length=smooth_window, polyorder=2, deriv=1)
    df['fcrit_dot'] = savgol_filter(df['fcrit_p'], window_length=smooth_window, polyorder=2, deriv=1)
    
    df['SpeedIndex'] = np.sqrt(df['beta_dot']**2 + df['fcrit_dot']**2)
    df['CoupleIndex'] = df['beta_dot'].rolling(window=couple_window).corr(df['fcrit_dot'])
    
    return df.dropna()

def plot_financial_cascade(df):
    """Generates a 2x2 plot for the full bubble lifecycle."""
    fig, axes = plt.subplots(2, 2, figsize=(20, 16))
    fig.suptitle('Threshold Dialectics of a Financial "Echo Bubble"', fontsize=28, y=0.98)
    
    # --- Define colors for phases ---
    phase_colors = {
        'P1_Lull': 'gray', 'P2_Emerge': 'lightblue', 'P3_Inflate': 'lightcoral',
        'P4_Fragility': 'lightsalmon', 'P5_Correct': 'lightskyblue', 'P6_Reset': 'lightgray'
    }

    # --- Panel 1: Lever Proxies ---
    ax = axes[0, 0]
    ax.plot(df.index, df['beta_p'], color='blue', lw=3, label=r'$\beta_p$ (Valuation Consensus)')
    ax.plot(df.index, df['fcrit_p'], color='red', lw=3, label=r'$F_{crit,p}$ (Margin Debt)')
    ax.set_title('Lever Proxies vs. Time', fontsize=18)
    ax.set_ylabel('Normalized Value', fontsize=16)
    ax.set_ylim(-0.05, 1.05)
    ax.legend(loc='best', fontsize=14)
    
    # --- Panel 2: Systemic Strain ---
    ax = axes[0, 1]
    ax.plot(df.index, df['strain_p'], color='black', lw=3, label=r'$\langle\Delta_P\rangle_p$ (Price Deviation)')
    ax.set_title('Systemic Strain Proxy vs. Time', fontsize=18)
    ax.set_ylabel('Strain Level', fontsize=16)
    ax.set_ylim(-0.05, 1.05)
    ax.legend(loc='best', fontsize=14)
         
    # --- Panel 3: TD Diagnostics ---
    ax = axes[1, 0]
    ax.plot(df.index, df['SpeedIndex'], color='darkorange', lw=3, label='Speed Index')
    ax.plot(df.index, df['CoupleIndex'], color='purple', lw=3, label='Couple Index')
    ax.axhline(0, color='gray', linestyle=':', lw=2)
    ax.set_title('TD Diagnostics vs. Time', fontsize=18)
    ax.set_ylabel('Index Value', fontsize=16)
    ax.set_ylim(-1.05, 1.05)
    ax.legend(loc='best', fontsize=14)
    
    # --- Panel 4: S/C Diagnostic Plane ---
    ax = axes[1, 1]
    # Risk zones
    ax.axvspan(0.04, 0.08, color='yellow', alpha=0.3)
    ax.axvspan(0.08, 0.2, color='red', alpha=0.3)
    ax.axhspan(0.5, 1.05, color='red', alpha=0.3)
    
    # Plot trajectory with color gradient over time
    points = np.array([df['SpeedIndex'], df['CoupleIndex']]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    norm = plt.Normalize(df.index.min(), df.index.max())
    lc = plt.cm.plasma(norm(df.index))
    
    for i in range(len(df)-1):
        ax.plot(df['SpeedIndex'][i:i+2], df['CoupleIndex'][i:i+2], color=lc[i], lw=2.5)

    ax.set_title('Trajectory on Diagnostic Plane', fontsize=18)
    ax.set_xlabel('Speed Index', fontsize=16)
    ax.set_ylabel('Couple Index', fontsize=16)
    ax.set_xlim(0, 0.15)
    ax.set_ylim(-1.05, 1.05)

    # Annotate phases on the trajectory plot
    phase_points = df.groupby('phase').first()
    for phase, row in phase_points.iterrows():
        ax.annotate(phase, (row['SpeedIndex'], row['CoupleIndex']),
                    textcoords="offset points", xytext=(-10, 10), ha='center',
                    arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=.2"),
                    bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="black", lw=1, alpha=0.7),
                    fontsize=12)

    # Add phase backgrounds to time series plots
    for ax_ts in [axes[0,0], axes[0,1], axes[1,0]]:
        for phase, color in phase_colors.items():
            phase_data = df[df['phase'] == phase]
            if not phase_data.empty:
                ax_ts.axvspan(phase_data.index.min(), phase_data.index.max(), color=color, alpha=0.2, zorder=-1)

    for ax_row in axes:
        for ax_col in ax_row:
            if ax_col is not axes[1, 1]:
                ax_col.set_xlabel('Week', fontsize=16)
            ax_col.tick_params(axis='both', which='major', labelsize=14)

    
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig("financial_bubble_dashboard.png", dpi=300, bbox_inches='tight')
    plt.show()

# --- NEW FUNCTION FOR STATISTICAL ANALYSIS ---
def run_statistical_analysis(df):
    """Runs and prints statistical tests to support the chapter's narrative."""
    print("\n" + "="*80)
    print("STATISTICAL ANALYSIS: FINANCIAL 'ECHO BUBBLE'")
    print("="*80)
    
    # Isolate data for the key phases
    lull_data = df[df['phase'] == 'P1_Lull'].dropna()
    inflate_data = df[df['phase'] == 'P3_Inflate'].dropna()

    # --- Test 1: Is SpeedIndex significantly higher in P3 vs P1? ---
    print("\n[H1] Testing if SpeedIndex is higher in Inflate Phase (P3) vs. Lull Phase (P1)...")
    speed_stat, speed_p = stats.mannwhitneyu(inflate_data['SpeedIndex'], lull_data['SpeedIndex'], alternative='greater')
    print(f"   Mann-Whitney U Test: U-statistic = {speed_stat:.2f}, p-value = {speed_p:.4e}")
    if speed_p < 0.05:
        print("   ✅ RESULT: The difference is statistically significant. SpeedIndex is higher in P3.")
    else:
        print("   ❌ RESULT: The difference is not statistically significant.")

    # --- Test 2: Is CoupleIndex significantly higher in P3 vs P1? ---
    print("\n[H2] Testing if CoupleIndex is higher in Inflate Phase (P3) vs. Lull Phase (P1)...")
    couple_stat, couple_p = stats.mannwhitneyu(inflate_data['CoupleIndex'], lull_data['CoupleIndex'], alternative='greater')
    print(f"   Mann-Whitney U Test: U-statistic = {couple_stat:.2f}, p-value = {couple_p:.4e}")
    if couple_p < 0.05:
        print("   ✅ RESULT: The difference is statistically significant. CoupleIndex is higher in P3.")
    else:
        print("   ❌ RESULT: The difference is not statistically significant.")

    # --- Test 3: Is there a strong positive correlation between levers in P3? ---
    print("\n[H3] Testing for positive correlation between beta_p and fcrit_p during P3_Inflate...")
    corr, p_val = stats.pearsonr(inflate_data['beta_p'], inflate_data['fcrit_p'])
    print(f"   Pearson Correlation: coefficient = {corr:.3f}, p-value = {p_val:.4e}")
    if p_val < 0.05 and corr > 0:
        print("   ✅ RESULT: A statistically significant positive correlation exists.")
    else:
        print("   ❌ RESULT: No significant positive correlation found.")
    print("="*80 + "\n")


# --- Main Execution ---
if __name__ == "__main__":
    # 1. Instantiate and run the simulation
    bubble_sim = FinancialMarketSystem(n_weeks=70, random_seed=101)
    history_df = bubble_sim.run()

    # 2. Calculate TD diagnostics
    full_df = calculate_td_diagnostics(history_df, couple_window=10)

    # 3. Plot the full lifecycle
    plot_financial_cascade(full_df)

    # 4. Run and print statistical tests <-- ADD THIS CALL
    run_statistical_analysis(full_df)



""" 
Example output:

================================================================================
STATISTICAL ANALYSIS: FINANCIAL 'ECHO BUBBLE'
================================================================================

[H1] Testing if SpeedIndex is higher in Inflate Phase (P3) vs. Lull Phase (P1)...
   Mann-Whitney U Test: U-statistic = 108.00, p-value = 1.7830e-01
   ❌ RESULT: The difference is not statistically significant.

[H2] Testing if CoupleIndex is higher in Inflate Phase (P3) vs. Lull Phase (P1)...
   Mann-Whitney U Test: U-statistic = 144.00, p-value = 5.1840e-03
   ✅ RESULT: The difference is statistically significant. CoupleIndex is higher in P3.

[H3] Testing for positive correlation between beta_p and fcrit_p during P3_Inflate...
   Pearson Correlation: coefficient = 0.950, p-value = 3.6530e-13
   ✅ RESULT: A statistically significant positive correlation exists.
================================================================================


"""