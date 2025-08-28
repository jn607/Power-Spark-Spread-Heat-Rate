"""Core calculations and plotting for power spark‑spread analysis.

This script loads simple price data from a CSV, computes gross and clean spark
spreads, classifies regimes based on percentiles, performs a small scenario
analysis and a toy rules‑based backtest, and writes out a chart and summary
files.  It avoids external data sources and keeps logic readable and
vectorised where possible.

Run from the repository root with::

    python src/spark_spread.py --config config.yaml

All output files are written into the ``outputs/`` directory adjacent to the
configuration file.
"""

import argparse
import os
from typing import Dict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yaml


def load_data(csv_path: str, lookback_years: int) -> pd.DataFrame:
    """Load the input CSV, drop missing values and restrict to the lookback window.

    Parameters
    ----------
    csv_path : str
        Path to the CSV file with columns date, power_eur_mwh, gas_eur_mwh, eua_eur_t.
    lookback_years : int
        Number of years to retain from the end of the series.

    Returns
    -------
    DataFrame
        Cleaned and filtered dataset sorted by date.
    """
    df = pd.read_csv(csv_path, parse_dates=['date'])
    df = df.dropna().sort_values('date').reset_index(drop=True)
    if lookback_years:
        # Determine cut‑off date using calendar years; use 365 days per year approximation.
        cutoff = df['date'].max() - pd.DateOffset(years=lookback_years)
        df = df[df['date'] >= cutoff].reset_index(drop=True)
    return df


def compute_spreads(df: pd.DataFrame, heat_rate: float, emission_factor: float) -> pd.DataFrame:
    """Compute gross and clean spark spreads.

    Gross spark = power – heat_rate × gas.
    Clean spark = gross spark – emission_factor × EUA.

    Parameters
    ----------
    df : DataFrame
        Input data with power, gas and EUA columns.
    heat_rate : float
        Gas consumption per MWh of power produced.
    emission_factor : float
        CO₂ emissions per MWh of power.

    Returns
    -------
    DataFrame
        DataFrame with additional columns gross_spark and clean_spark.
    """
    df = df.copy()
    df['gross_spark'] = df['power_eur_mwh'] - heat_rate * df['gas_eur_mwh']
    df['clean_spark'] = df['gross_spark'] - emission_factor * df['eua_eur_t']
    return df


def compute_percentiles(df: pd.DataFrame) -> pd.DataFrame:
    """Compute percentile ranks of the clean spark spread.

    Percentile is calculated across the entire lookback window (relative rank).

    Returns
    -------
    DataFrame
        DataFrame with a new column percentile between 0 and 1.
    """
    df = df.copy()
    df['percentile'] = df['clean_spark'].rank(pct=True)
    return df


def scenario_grid(df: pd.DataFrame, config: Dict) -> pd.DataFrame:
    """Construct a grid of scenario deltas and recompute clean spark for each.

    The last row of the input data provides the base prices.  For each
    combination of deltas on power, gas and EUA defined in the config, a new
    clean spark value is calculated.

    Returns
    -------
    DataFrame
        Grid with columns d_power, d_gas, d_eua and clean_spark.
    """
    base = df.iloc[-1]
    heat_rate = config['heat_rate_mwh_gas_per_mwh_power']
    emission_factor = config['emission_factor_tco2_per_mwh_power']
    deltas_p = config['scenario']['d_power_eur_mwh']
    deltas_g = config['scenario']['d_gas_eur_mwh']
    deltas_e = config['scenario']['d_eua_eur_t']
    rows = []
    for dp in deltas_p:
        for dg in deltas_g:
            for de in deltas_e:
                new_clean = (base['power_eur_mwh'] + dp) - heat_rate * (base['gas_eur_mwh'] + dg) - emission_factor * (base['eua_eur_t'] + de)
                rows.append({'d_power_eur_mwh': dp,
                             'd_gas_eur_mwh': dg,
                             'd_eua_eur_t': de,
                             'clean_spark': new_clean})
    return pd.DataFrame(rows)


def beta_estimation(df: pd.DataFrame) -> np.ndarray:
    """Estimate simple betas of clean spread changes vs changes in power, gas and EUA.

    Uses ordinary least squares on daily differences without intercept.

    Returns
    -------
    ndarray
        Array of three betas corresponding to power, gas and EUA legs.
    """
    d_power = df['power_eur_mwh'].diff().values[1:]
    d_gas = df['gas_eur_mwh'].diff().values[1:]
    d_eua = df['eua_eur_t'].diff().values[1:]
    d_clean = df['clean_spark'].diff().values[1:]
    X = np.column_stack([d_power, d_gas, d_eua])
    # Remove rows with any NaN (should not occur) to avoid singular matrix
    mask = ~np.isnan(X).any(axis=1) & ~np.isnan(d_clean)
    X = X[mask]
    y = d_clean[mask]
    if len(y) == 0:
        return np.array([0.0, 0.0, 0.0])
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    return beta


def discontinuity_flags(df: pd.DataFrame) -> pd.Series:
    """Flag days where the clean spread moves more than twice its recent volatility.

    Volatility is measured as the rolling 60‑day standard deviation of daily changes.

    Returns
    -------
    Series
        Boolean series indicating discontinuity events.
    """
    d_clean = df['clean_spark'].diff()
    rolling_std = d_clean.rolling(60).std()
    flags = (d_clean.abs() > 2 * rolling_std)
    # Ensure any NaN periods return False
    return flags.fillna(False)


def backtest(df: pd.DataFrame, entry_pct: float, exit_pct: float, atr_window: int, stop_mult: float) -> pd.DataFrame:
    """Run a simple percentile‑based long/flat strategy on the clean spark spread.

    Parameters
    ----------
    df : DataFrame
        Data containing clean_spark and percentile columns.
    entry_pct : float
        Percentile threshold for entering long (cross up through this level).
    exit_pct : float
        Percentile threshold for exiting to flat (cross down through this level).
    atr_window : int
        Window length in days for the ATR calculation (uses absolute d_clean).
    stop_mult : float
        Multiple of ATR used as a stop‑loss on drawdown from entry.

    Returns
    -------
    DataFrame
        Single‑row DataFrame with summary metrics.
    """
    d_clean = df['clean_spark'].diff().fillna(0)
    atr = d_clean.abs().rolling(atr_window).mean().fillna(0)
    position = 0
    entry_price = 0.0
    entry_atr = 0.0
    trades = []
    positions = np.zeros(len(df), dtype=int)
    for i in range(1, len(df)):
        prev_pct = df['percentile'].iat[i - 1]
        curr_pct = df['percentile'].iat[i]
        if position == 0:
            # Entry condition: crossing up through entry_pct
            if prev_pct <= entry_pct and curr_pct > entry_pct:
                position = 1
                entry_price = df['clean_spark'].iat[i]
                entry_atr = atr.iat[i]
        else:
            # Stop condition: drawdown > stop_mult × entry_atr
            drawdown = entry_price - df['clean_spark'].iat[i]
            stop_hit = drawdown > stop_mult * entry_atr if entry_atr > 0 else False
            # Exit condition: crossing down through exit_pct
            cross_down = prev_pct >= exit_pct and curr_pct < exit_pct
            if stop_hit or cross_down:
                exit_price = df['clean_spark'].iat[i]
                trades.append(exit_price - entry_price)
                position = 0
                entry_price = 0.0
                entry_atr = 0.0
        positions[i] = position
    # Daily P&L: change in clean spread times position
    daily_pl = d_clean * positions
    # Summary metrics
    trades_arr = np.array(trades)
    num_trades = int(len(trades_arr))
    hit_rate = float((trades_arr > 0).mean()) if num_trades else 0.0
    avg_win = float(trades_arr[trades_arr > 0].mean()) if (trades_arr > 0).any() else 0.0
    avg_loss = float(trades_arr[trades_arr <= 0].mean()) if (trades_arr <= 0).any() else 0.0
    cum_pl = daily_pl.cumsum()
    running_max = np.maximum.accumulate(cum_pl)
    max_drawdown = float((running_max - cum_pl).max()) if len(cum_pl) else 0.0
    sharpe = float(daily_pl.mean() / daily_pl.std()) if daily_pl.std() != 0 else 0.0
    return pd.DataFrame({
        'trades': [num_trades],
        'hit_rate': [hit_rate],
        'avg_win': [avg_win],
        'avg_loss': [avg_loss],
        'max_drawdown': [max_drawdown],
        'sharpe': [sharpe]
    })


def plot_series(df: pd.DataFrame, pct_low: float, pct_high: float, flags: pd.Series, annotate_today: bool, out_path: str) -> None:
    """Create a two‑panel plot of the clean spark spread and discontinuity markers.

    Parameters
    ----------
    df : DataFrame
        Data containing date and clean_spark columns.
    pct_low, pct_high : float
        Percentile levels defining the neutral band.
    flags : Series
        Boolean flags marking discontinuity events.
    annotate_today : bool
        Whether to annotate the latest value on the plot.
    out_path : str
        Output filename for the PNG chart.
    """
    q_low = df['clean_spark'].quantile(pct_low)
    q_high = df['clean_spark'].quantile(pct_high)
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6), sharex=True,
                                   gridspec_kw={'height_ratios': [3, 1]})
    # Upper panel: clean spark spread
    ax1.plot(df['date'], df['clean_spark'], label='Clean spark spread', color='tab:blue')
    ax1.fill_between(df['date'], q_low, q_high, color='gray', alpha=0.2, label='Neutral band')
    if annotate_today and not df.empty:
        last = df.iloc[-1]
        ax1.scatter(last['date'], last['clean_spark'], color='red', zorder=5)
        text = (f"{last['clean_spark']:.2f} €/MWh\n"
                f"Pctile: {last['percentile'] * 100:.1f}%\n"
                f"Regime: {last['regime']}")
        ax1.annotate(text, (last['date'], last['clean_spark']),
                    xytext=(10, 0), textcoords='offset points', va='center', fontsize=8,
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    ax1.set_ylabel('Clean spark (€/MWh)')
    ax1.set_title('Clean spark spread')
    ax1.legend(loc='upper left', fontsize=8)
    # Lower panel: discontinuity markers
    ax2.scatter(df['date'][flags], np.zeros(flags.sum()), marker='o', color='red', s=10)
    ax2.set_yticks([])
    ax2.set_ylabel('Discontinuity', labelpad=15)
    ax2.set_xlabel('Date')
    fig.autofmt_xdate()
    plt.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def write_note(out_path: str, today: pd.Series, betas: np.ndarray, df: pd.DataFrame) -> None:
    """Write a concise markdown note summarising the latest state and sensitivities.

    Parameters
    ----------
    out_path : str
        Destination file path.
    today : Series
        The most recent row of the data.
    betas : ndarray
        Regression betas for power, gas and EUA legs.
    df : DataFrame
        The entire lookback data, used for context (e.g. length).
    """
    legs = ['power', 'gas', 'EUA']
    dom_leg = legs[int(np.argmax(np.abs(betas)))] if betas.size else 'power'
    value = today['clean_spark']
    pctile = today['percentile'] * 100
    regime = today['regime']
    lines = []
    lines.append(f"**Overview:** This chart shows the clean spark spread (€/MWh) for a gas‑fired power plant net of carbon costs over the last {len(df)} days. A shaded band marks the 30–70th percentile range, and red markers flag days when the spread moved more than twice its recent volatility.")
    lines.append("\n**Drivers:** The clean spark spread is driven by the electricity price, gas price (via the heat‑rate) and EUA price (via the emission factor). Fundamental factors such as outages, load, interconnector flows and hydro/wind availability can influence these prices.")
    lines.append(f"\n**Today:** The latest clean spark spread is **{value:.2f} €/MWh**, placing it at the **{pctile:.1f}th percentile** of the lookback. Current regime: **{regime}**.")
    lines.append(f"\n**Sensitivity:** A simple regression of daily changes in clean spark on changes in each leg suggests the spread is most sensitive to **{dom_leg}** prices.")
    lines.append("\n**Bias:** stay constructive spark if clean spread >70th percentile and EUA steady. **Invalidator:** drop back <50th percentile or gas rally +€10/MWh without power following.")
    with open(out_path, 'w') as f:
        f.write('\n'.join(lines))


def main(config_path: str) -> None:
    """Execute the full workflow using the provided configuration file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    # Derive data path relative to the config file
    repo_dir = os.path.dirname(os.path.abspath(config_path))
    data_path = os.path.join(repo_dir, 'data', 'sample_power.csv')
    outputs_dir = os.path.join(repo_dir, 'outputs')
    os.makedirs(outputs_dir, exist_ok=True)
    # Load and process data
    df = load_data(data_path, config.get('lookback_years', 3))
    df = compute_spreads(df, config['heat_rate_mwh_gas_per_mwh_power'], config['emission_factor_tco2_per_mwh_power'])
    df = compute_percentiles(df)
    # Classify regimes
    pct_high = config.get('pct_high', 0.7)
    pct_low = config.get('pct_low', 0.3)
    df['regime'] = np.where(df['percentile'] > pct_high, 'tight',
                            np.where(df['percentile'] < pct_low, 'loose', 'neutral'))
    # Discontinuities
    flags = discontinuity_flags(df)
    # Scenario grid and betas
    scen_df = scenario_grid(df, config)
    scen_df.to_csv(os.path.join(outputs_dir, 'spark_spread_scenarios.csv'), index=False)
    betas = beta_estimation(df)
    # Backtest
    bt_cfg = config.get('backtest', {})
    metrics_df = backtest(df,
                          bt_cfg.get('entry_percentile', 0.7),
                          bt_cfg.get('exit_percentile', 0.5),
                          bt_cfg.get('atr_window_days', 20),
                          bt_cfg.get('stop_atr', 1.0))
    metrics_df.to_csv(os.path.join(outputs_dir, 'backtest_metrics.csv'), index=False)
    # Plot
    plot_cfg = config.get('plot', {})
    plot_series(df, pct_low, pct_high, flags, plot_cfg.get('annotate_today', True),
                os.path.join(outputs_dir, 'spark_spread.png'))
    # Note
    write_note(os.path.join(outputs_dir, 'spark_spread_note.md'), df.iloc[-1], betas, df)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Compute spark spreads and related analytics.')
    parser.add_argument('--config', required=True, help='Path to YAML configuration file.')
    args = parser.parse_args()
    main(args.config)