"""CAPM analytics with aligned periodic simple returns."""
import numpy as np
import pandas as pd

TRADING_DAYS = 252

def analyze(stock_prices, market_prices, risk_free_pct=6.0, rolling_window=60):
    if rolling_window < 5:
        raise ValueError('Rolling window must be at least 5')
    if not np.isfinite(risk_free_pct):
        raise ValueError('Risk-free rate must be finite')
    prices = pd.concat([pd.Series(stock_prices, dtype=float).rename('Stock'),
                        pd.Series(market_prices, dtype=float).rename('Market')], axis=1).dropna().sort_index()
    if len(prices) < max(rolling_window + 2, 12) or (prices <= 0).any().any() or not np.isfinite(prices.to_numpy()).all():
        raise ValueError('Not enough aligned, positive, finite prices')
    ret = prices.pct_change().dropna()
    if ret['Market'].var(ddof=1) <= 1e-16:
        raise ValueError('Market return variance is zero')
    beta = float(ret['Stock'].cov(ret['Market']) / ret['Market'].var())
    intercept = float(ret['Stock'].mean() - beta * ret['Market'].mean())
    correlation = float(ret['Stock'].corr(ret['Market']))
    stock_ann = float((1 + ret['Stock']).prod() ** (TRADING_DAYS / len(ret)) - 1)
    market_ann = float((1 + ret['Market']).prod() ** (TRADING_DAYS / len(ret)) - 1)
    rf = risk_free_pct / 100
    expected = rf + beta * (market_ann - rf)
    # Historical realized alpha uses arithmetic daily excess returns, annualized.
    jensen_alpha = (ret['Stock'].mean() - rf / TRADING_DAYS - beta * (ret['Market'].mean() - rf / TRADING_DAYS)) * TRADING_DAYS
    ret['Rolling beta'] = ret['Stock'].rolling(rolling_window).cov(ret['Market']) / ret['Market'].rolling(rolling_window).var()
    summary = {'Beta': beta, 'CAPM expected return': expected, 'Historical stock CAGR': stock_ann,
               'Historical market CAGR': market_ann, 'Jensen alpha (annualized)': float(jensen_alpha),
               'Correlation': correlation, 'R-squared': correlation**2,
               'Annualized stock volatility': float(ret['Stock'].std() * np.sqrt(TRADING_DAYS)),
               'Daily regression intercept': intercept}
    return ret, summary
