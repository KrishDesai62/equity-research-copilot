"""Evaluate dated recorded agent actions against fixed risk/news and buy-and-hold.

Input returns must be total returns INCLUDING dividends and delisting outcomes,
from each executable start to exit. This engine validates timing and arithmetic;
it cannot certify a vendor's universe or return completeness.
"""
import argparse
from hashlib import sha256
import json
from pathlib import Path
from urllib.parse import urlparse

import numpy as np
import pandas as pd

from src.backtest.mechanical_baseline import positions
from src.data_layer.research_dataset import write_json


def evaluate(panel, *, cost_bps=10., max_risk=.1):
    data = panel.copy()
    dates = ['decision_at', 'available_at', 'model_trained_through', 'entry_at', 'exit_at']
    for column in dates:
        data[column] = pd.to_datetime(data[column], utc=True, errors='raise')
    if data.empty or data[dates + ['ticker', 'agent_action', 'evidence_url']].isna().any().any():
        raise ValueError('Backtest requires complete dated actions and evidence')
    if data.duplicated(['decision_at', 'ticker']).any() or not set(data.agent_action) <= {'buy', 'hold', 'sell'}:
        raise ValueError('Duplicate decision/company or invalid agent action')
    if ((data.available_at > data.decision_at).any() or (data.model_trained_through >= data.decision_at).any()
            or (data.entry_at <= data.decision_at).any() or (data.exit_at <= data.entry_at).any()):
        raise ValueError('Look-ahead or invalid execution dates')
    for url in data.evidence_url:
        parsed = urlparse(str(url))
        if parsed.scheme != 'https' or not parsed.netloc:
            raise ValueError('Agent decisions need dated public source citations')
    returns = data.total_return.to_numpy(float)
    if not np.isfinite(returns).all() or (returns < -1).any() or not np.isfinite(cost_bps) or not 0 <= cost_bps <= 1000:
        raise ValueError('Invalid total returns or transaction costs')
    universe = sorted(data.ticker.unique())
    n = len(universe)
    previous = {key: np.zeros(n) for key in ['agent', 'mechanical', 'buy_hold']}
    held, records, last_exit = np.zeros(n), [], None
    for decision, group in data.groupby('decision_at', sort=True):
        group = group.set_index('ticker').sort_index()
        if list(group.index) != universe or group.entry_at.nunique() != 1 or group.exit_at.nunique() != 1:
            raise ValueError('Supply a complete fixed-universe panel with common execution intervals; never drop failed firms')
        if last_exit is not None and decision < last_exit:
            raise ValueError('Overlapping strategy return intervals')
        last_exit = group.exit_at.iloc[0]
        action = group.agent_action.to_numpy()
        held = np.where(action == 'buy', 1., np.where(action == 'sell', 0., held))
        weights = {'agent': held / n,
            'mechanical': positions(group.bankruptcy_probability, group.news_sentiment, max_risk=max_risk) / n,
            'buy_hold': np.ones(n) / n if not records else previous['buy_hold'].copy()}
        realized = group.total_return.to_numpy(float)
        row = {'decision_at': decision.isoformat(), 'exit_at': last_exit.isoformat()}
        for name, w in weights.items():
            if not np.isfinite(w).all() or (w < 0).any() or w.sum() > 1 + 1e-10:
                raise ValueError('Invalid strategy weights')
            turnover = float(np.abs(w - previous[name]).sum())
            fee = turnover * cost_bps / 10000
            gross = float(w @ realized)
            row[name + '_return'] = (1 - fee) * (1 + gross) - 1
            row[name + '_turnover'] = turnover
            previous[name] = w * (1 + realized) / (1 + gross) if gross > -1 else np.zeros(n)
        records.append(row)
    table = pd.DataFrame(records)
    results = {}
    for name in previous:
        equity = np.r_[1., np.cumprod(1 + table[name + '_return'].to_numpy())]
        results[name] = {'total_return': float(equity[-1] - 1),
            'max_drawdown': float((equity / np.maximum.accumulate(equity) - 1).min()),
            'total_turnover': float(table[name + '_turnover'].sum())}
    return {'periods': len(table), 'companies': n, 'cost_bps_per_traded_dollar': cost_bps,
            'mechanical_max_risk': max_risk, 'strategies': results, 'period_results': records,
            'validated_agent_profitability': False,
            'limitations': ['Requires independently reviewed point-in-time universe, total returns and actual recorded agent decisions.',
                'Fixed supplied universe; not a historical index reconstruction. No significance claim or strategy optimization.',
                'Uninvested capital earns zero; taxes, borrow, market impact and final liquidation costs are excluded.']}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--panel', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--cost-bps', type=float, default=10.)
    args = parser.parse_args()
    report = evaluate(pd.read_csv(args.panel), cost_bps=args.cost_bps)
    report['input_panel_sha256'] = sha256(args.panel.read_bytes()).hexdigest()
    report['engine_sha256'] = sha256(Path(__file__).read_bytes()).hexdigest()
    if args.output.exists():
        raise FileExistsError('Keep existing strategy evidence immutable; choose a new output')
    write_json(args.output, report)
    print(json.dumps(report, indent=2))
