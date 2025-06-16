# echo_bubble_19A.py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
import seaborn as sns
import scipy.stats as stats
# Set a professional plotting theme with larger text
plt.rcParams.update({
    "font.size": 24,
    "axes.titlesize": 24,
    "axes.labelsize": 24,
    "legend.fontsize": 24,
    "xtick.labelsize": 14,
    "ytick.labelsize": 14
})

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
    """
    Generates a final, PUBLICATION-QUALITY 2x2 plot for the bubble lifecycle,
    using a unified color scheme and a high-contrast "cased line" for clarity.
    """
    # Use the seaborn 'whitegrid' style for consistency with the lake figure
    
    fig, axes = plt.subplots(2, 2, figsize=(20, 16))
    fig.suptitle('Threshold Dialectics of a Financial "Echo Bubble"', fontsize=28, y=0.98)

    # --- FINAL, REFINED Narrative and Styling Configuration ---
    # Unified color palette with more distinct colors for each narrative phase
    NARRATIVE_COLORS = {
        'Pre-Bubble (P1-P2)': '#B0C4DE',  # Light Steel Blue
        'Inflation (P3-P4)': '#FF6347',    # Tomato Red (More vibrant)
        'Correction (P5-P6)': '#87CEEB', # Sky Blue (Distinct from Pre-Bubble)
    }
    # Map individual phases to their narrative group color
    PHASE_TO_NARRATIVE_COLOR = {
        'P1_Lull': NARRATIVE_COLORS['Pre-Bubble (P1-P2)'],
        'P2_Emerge': NARRATIVE_COLORS['Pre-Bubble (P1-P2)'],
        'P3_Inflate': NARRATIVE_COLORS['Inflation (P3-P4)'],
        'P4_Fragility': NARRATIVE_COLORS['Inflation (P3-P4)'],
        'P5_Correct': NARRATIVE_COLORS['Correction (P5-P6)'],
        'P6_Reset': NARRATIVE_COLORS['Correction (P5-P6)'],
    }
    LINE_COLORS = {'beta_p': 'blue', 'fcrit_p': 'red', 'strain_p': 'black',
                   'SpeedIndex': 'darkorange', 'CoupleIndex': 'purple'}

    # --- Identify Key Event Time for Annotation ---
    correction_start_week = df[df['phase'] == 'P5_Correct'].index.min()

    # --- Time-Series Plots (Panels 1, 2, 3) ---
    # Panel 1: Lever Proxies
    ax = axes[0, 0]
    ax.plot(df.index, df['beta_p'], color=LINE_COLORS['beta_p'], lw=3, label=r'$\beta_p$ (Valuation Consensus)')
    ax.plot(df.index, df['fcrit_p'], color=LINE_COLORS['fcrit_p'], lw=3, label=r'$F_{crit,p}$ (Margin Debt)')
    ax.axvline(correction_start_week, color='black', linestyle='--', lw=2.5, label='Correction Start')
    ax.set_title('Lever Proxies vs. Time')
    ax.set_ylabel('Normalized Value')
    ax.legend(loc='upper left')

    # Panel 2: Systemic Strain
    ax = axes[0, 1]
    ax.plot(df.index, df['strain_p'], color=LINE_COLORS['strain_p'], lw=3, label=r'$\langle\Delta_P\rangle_p$ (Price Deviation)')
    ax.axvline(correction_start_week, color='black', linestyle='--', lw=2.5)
    ax.set_title('Systemic Strain Proxy vs. Time')
    ax.legend(loc='upper left')

    # Panel 3: TD Diagnostics
    ax = axes[1, 0]
    ax.plot(df.index, df['SpeedIndex'], color=LINE_COLORS['SpeedIndex'], lw=3, label='Speed Index')
    ax.plot(df.index, df['CoupleIndex'], color=LINE_COLORS['CoupleIndex'], lw=3, label='Couple Index')
    ax.axhline(0, color='gray', linestyle=':', lw=2)
    ax.axvline(correction_start_week, color='black', linestyle='--', lw=2.5)
    ax.set_title('TD Diagnostics vs. Time')
    ax.set_ylabel('Index Value')
    ax.legend(loc='best')

    # --- Panel 4: Trajectory on Diagnostic Plane (FINAL POLISHED VERSION) ---
    ax = axes[1, 1]
    # Risk zones
    ax.axvspan(0.06, 0.1, color='yellow', alpha=0.3, zorder=-2)
    ax.axvspan(0.1, 0.2, color='red', alpha=0.3, zorder=-2)
    ax.axhspan(0.7, 1.05, color='red', alpha=0.3, zorder=-2)
    ax.axhspan(-1.05, -0.7, color='red', alpha=0.3, zorder=-2)

    # Plot trajectories with special handling for the Inflation phase
    for group_name, color in NARRATIVE_COLORS.items():
        phase_list = [p for p, c in PHASE_TO_NARRATIVE_COLOR.items() if c == color]
        group_data = df[df['phase'].isin(phase_list)]
        
        # --- THIS IS THE KEY IMPROVEMENT ---
        if group_name == 'Inflation (P3-P4)':
            # Plot a thicker black line underneath to create a "cased line" effect
            ax.plot(group_data['SpeedIndex'], group_data['CoupleIndex'],
                    color='black', lw=5, zorder=3)
            # Plot the main colored line on top
            ax.plot(group_data['SpeedIndex'], group_data['CoupleIndex'],
                    color=color, lw=3, label=group_name, zorder=4)
        else:
            # Plot other phases normally
            ax.plot(group_data['SpeedIndex'], group_data['CoupleIndex'],
                    color=color, lw=4, label=group_name, zorder=2) # zorder=2 to be behind markers
            
    # Add key event markers with enhanced visibility
    start_point = df.iloc[0]
    correction_point = df.loc[correction_start_week]
    ax.scatter(start_point['SpeedIndex'], start_point['CoupleIndex'],
               marker='o', s=250, color='blue', ec='black', lw=1.5, zorder=5, label='Start')
    ax.scatter(correction_point['SpeedIndex'], correction_point['CoupleIndex'],
               marker='X', s=300, color='white', ec='black', lw=1.5, zorder=5, label='Correction Start') # White 'X' for max contrast

    ax.set_title('Trajectory on Diagnostic Plane')
    ax.set_xlabel('Speed Index')
    ax.set_ylabel('Couple Index')
    ax.set_xlim(0, 0.15)
    ax.set_ylim(-1.05, 1.05)
    ax.legend(title="Trajectory & Events", loc='lower right')

    # Apply UNIFIED Narrative Phase Backgrounds to all Time-Series Plots
    for ax_ts in [axes[0, 0], axes[0, 1], axes[1, 0]]:
        for phase_name, color in PHASE_TO_NARRATIVE_COLOR.items():
            phase_data = df[df['phase'] == phase_name]
            if not phase_data.empty:
                ax_ts.axvspan(phase_data.index.min(), phase_data.index.max(),
                              color=color, alpha=0.2, zorder=-1)

    # Final Polish on all Axes
    for ax_row in axes:
        for ax_col in ax_row:
            if ax_col is not axes[1, 1]:
                ax_col.set_xlabel('Week')
                # Remove y-axis labels for strain and index plots for cleaner look if desired
                # (Optional, but can improve dashboard feel)
                if ax_col in [axes[0, 1], axes[1, 0]]:
                    ax_col.set_ylabel('')
            ax_col.tick_params(axis='both', which='major', labelsize=12)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig("financial_bubble_dashboard.png", dpi=350, bbox_inches='tight')
    plt.show()


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