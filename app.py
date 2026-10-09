"""Interactive CAPM and beta research dashboard."""
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from analytics import analyze

st.set_page_config(page_title='CAPM & Stock Beta Analyzer', page_icon='📊', layout='wide')
st.title('📊 CAPM & Stock Beta Analyzer')
st.caption('Explore systematic risk, CAPM-implied returns and rolling beta using historical prices.')
with st.sidebar:
    stock = st.text_input('Stock ticker', 'RELIANCE.NS').strip().upper()
    benchmark = st.text_input('Market benchmark ticker', '^NSEI').strip().upper()
    start = st.date_input('Start date', pd.Timestamp('2020-01-01'))
    end = st.date_input('End date', pd.Timestamp.today())
    rf = st.number_input('Annual risk-free rate (%)', value=6.0, step=0.25)
    window = st.slider('Rolling beta window (trading days)', 20, 252, 60)
    upload = st.file_uploader('Optional offline CSV: Date,Stock,Market', type='csv')
    run = st.button('Analyze', type='primary')

@st.cache_data(ttl=3600)
def download(symbol, start_date, end_date):
    import yfinance as yf
    raw = yf.download(symbol, start=start_date, end=end_date, auto_adjust=True, progress=False)
    if raw.empty:
        raise ValueError(f'No prices returned for {symbol}')
    close = raw['Close']
    if isinstance(close, pd.DataFrame):
        close = close.iloc[:, 0]
    return close.rename(symbol)

if run:
    try:
        if start >= end:
            raise ValueError('Start date must be earlier than end date')
        if upload is not None:
            df = pd.read_csv(upload)
            if not {'Date', 'Stock', 'Market'}.issubset(df.columns):
                raise ValueError('CSV must contain Date, Stock and Market columns')
            df['Date'] = pd.to_datetime(df['Date'])
            df = df.sort_values('Date').drop_duplicates('Date').set_index('Date')
            df = df.loc[str(start):str(end)]
            stock_prices, market_prices = df['Stock'], df['Market']
        else:
            if not stock or not benchmark:
                raise ValueError('Enter both tickers')
            stock_prices = download(stock, str(start), str(end))
            market_prices = download(benchmark, str(start), str(end))
        returns, m = analyze(stock_prices, market_prices, rf, window)
        c1, c2, c3, c4 = st.columns(4)
        c1.metric('Beta', f"{m['Beta']:.2f}")
        c2.metric('CAPM expected return', f"{m['CAPM expected return']:.2%}")
        c3.metric('Jensen alpha (annual)', f"{m['Jensen alpha (annualized)']:.2%}")
        c4.metric('R²', f"{m['R-squared']:.3f}")
        st.subheader('Stock vs. market daily returns')
        fig = px.scatter(returns.reset_index(), x='Market', y='Stock', trendline=None,
                         labels={'Market':'Market daily return', 'Stock':'Stock daily return'})
        x = returns['Market']
        fig.add_trace(go.Scatter(x=[x.min(), x.max()],
                                 y=[m['Daily regression intercept'] + m['Beta'] * x.min(),
                                    m['Daily regression intercept'] + m['Beta'] * x.max()],
                                 name='OLS regression', mode='lines'))
        st.plotly_chart(fig, use_container_width=True)
        fig2 = px.line(returns, y='Rolling beta', title='Rolling beta over time')
        st.plotly_chart(fig2, use_container_width=True)
        st.subheader('Metrics')
        st.dataframe(pd.DataFrame({'Metric':list(m.keys()),'Value':list(m.values())}), hide_index=True)
        st.download_button('Download daily returns and rolling beta',
                           returns.to_csv(index_label='Date'), 'capm_results.csv', 'text/csv')
        st.info('Historical statistical analysis, not an investment recommendation. CAPM is a model, '
                'not a forecast. The market risk premium here uses the benchmark’s historical CAGR; '
                'the risk-free rate is user-supplied. Daily returns are adjusted-price returns where available.')
    except Exception as exc:
        st.error(f'Analysis failed: {exc}')
else:
    st.info('Choose tickers or upload the sample CSV, then click Analyze.')
