# Streamlit & Giao diện
import streamlit as st

# Dữ liệu & xử lý dữ liệu
import pandas as pd
import numpy as np
import os

# Trực quan hóa
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import uuid
import yfinance as yf
from datetime import datetime, timedelta

# Phân tích thống kê & mô hình chuỗi thời gian
from scipy import stats
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.holtwinters import (
    SimpleExpSmoothing,
    Holt,
    ExponentialSmoothing
)
from statsmodels.tsa.arima.model import ARIMA

# Đánh giá mô hình
import joblib
from sklearn.metrics import mean_squared_error, mean_absolute_error
from tensorflow.keras.models import load_model


#Tiền sử lý vòng 2
#Chuyển đổi dữ liệu thành dạng float
def safe_float(x):
    """Safely convert Pandas Series or single values to float"""
    try:
        if isinstance(x, pd.Series):  # Kiểm tra nếu đầu vào là Pandas Series
            return float(x.iloc[0])  # Lấy giá trị đầu tiên và chuyển sang float
        return float(x)  # Chuyển trực tiếp sang float nếu là giá trị đơn
    except (ValueError, TypeError) as e:
        print(f"Warning: Cannot convert {x} to float. Error: {e}")
        return None  # Trả về None nếu lỗi


#Tham số cơ bản
def calculate_statistics(df, start_date=None, end_date=None):
    """Tính toán thống kê cho các cột Close, Open, High, Low, Volume trong khoảng thời gian cụ thể."""

    # Đảm bảo cột Date tồn tại và đúng định dạng
    if 'Date' in df.columns:
        df['Date'] = pd.to_datetime(df['Date']).dt.tz_localize(None)

        # Lọc theo khoảng ngày nếu có
        if start_date and end_date:
            start_date = pd.to_datetime(start_date)
            end_date = pd.to_datetime(end_date)
            df = df[(df['Date'] >= start_date) & (df['Date'] <= end_date)]

    # Giữ lại các cột mong muốn nếu tồn tại
    columns_to_keep = ['Close', 'Open', 'High', 'Low', 'Volume']
    available_cols = [col for col in columns_to_keep if col in df.columns]
    df_numeric = df[available_cols]

    # Tính thống kê cơ bản
    statistics = df_numeric.describe().to_dict()

    for col in df_numeric.columns:
        statistics[col]['Mode'] = df_numeric[col].mode()[0] if not df_numeric[col].mode().empty else np.nan
        statistics[col]['Sample Variance'] = df_numeric[col].var()
        statistics[col]['Kurtosis'] = df_numeric[col].kurt()
        statistics[col]['Skewness'] = df_numeric[col].skew()
        statistics[col]['Range'] = df_numeric[col].max() - df_numeric[col].min()
        statistics[col]['Sum'] = df_numeric[col].sum()

        # Confidence Interval 95%
        try:
            ci = stats.t.interval(
                0.95,
                len(df_numeric[col]) - 1,
                loc=np.mean(df_numeric[col]),
                scale=stats.sem(df_numeric[col])
            )
            statistics[col]['Confidence Interval (95%)'] = f"{ci[0]:.2f} - {ci[1]:.2f}"
        except Exception:
            statistics[col]['Confidence Interval (95%)'] = "N/A"

    # Trả về DataFrame thống kê
    stats_df = pd.DataFrame(statistics).transpose()
    return stats_df


#Biểu đồ Histogram Close
def plot_interactive_close_histogram(df, bins=30):
    if 'Close' not in df.columns:
        st.error("Dữ liệu không có cột 'Close'")
        return

    fig = px.histogram(
        df, 
        x='Close', 
        nbins=bins,
        labels={'Close': 'Giá đóng cửa'},
        opacity=0.85,
        color_discrete_sequence=['#1E90FF']
    )

    fig.update_layout(
        bargap=0.1,
        xaxis_title='Giá đóng cửa',
        yaxis_title='Tần suất xuất hiện',
        template='plotly_white',
        autosize=True,
        margin=dict(l=20, r=40, t=40, b=20),
        height=400,
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(family="Poppins, Roboto, sans-serif", size=16, color="#333333"),
        modebar_bgcolor='rgba(0,0,0,0)',       # Nền trong suốt cho Modebar
        modebar_activecolor='#007bff',         # Icon được chọn
        modebar_color='#333333'                # Màu mặc định của icon Modebar
    )

    # Xóa lưới cho sạch
    fig.update_xaxes(showgrid=False)
    fig.update_yaxes(showgrid=False)

    # Hiển thị biểu đồ
    st.plotly_chart(fig, use_container_width=True)


# Biểu đồ Histogram Volume
def plot_interactive_volume_histogram(df, bins=30):
    """
    Vẽ biểu đồ histogram tương tác cho chỉ số 'Volume' bằng Plotly.
    
    Parameters:
    - df (pd.DataFrame): Dữ liệu chứa cột 'Volume'.
    - bins (int): Số lượng cột trong histogram.
    """
    if 'Volume' not in df.columns:
        st.error("Dữ liệu không có cột 'Volume'")
        return

    fig = px.histogram(
        df, 
        x='Volume', 
        nbins=bins,
        labels={'Volume': 'Số lượng giao dịch'},
        opacity=0.85,
        color_discrete_sequence=['#1E90FF']
    )

    fig.update_layout(
        bargap=0.1,
        xaxis_title='Số lượng giao dịch',
        yaxis_title='Tần suất xuất hiện',
        template='plotly_white',
        autosize=True,
        margin=dict(l=20, r=0, t=40, b=20),
        height=400,
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(family="Poppins, Roboto, sans-serif", size=16, color="#333333"),
        modebar_bgcolor='rgba(0,0,0,0)',
        modebar_activecolor='#007bff',
        modebar_color='#333333'
    )

    fig.update_xaxes(showgrid=False)
    fig.update_yaxes(showgrid=False)

    st.plotly_chart(fig, use_container_width=True)

# Biểu đồ Histogram Tỉ lệ tăng trưởng
def plot_growth_histogram(df_filtered):
    """
    Vẽ biểu đồ histogram cho tỉ lệ tăng trưởng với cấu hình giống histogram Close.
    """
    df_filtered['Return (%)'] = df_filtered['Close'].pct_change() * 100
    df_returns = df_filtered.dropna(subset=['Return (%)'])

    fig = px.histogram(
        df_returns,
        x='Return (%)',
        nbins=50,
        opacity=0.85,
        color_discrete_sequence=['#1E90FF']
    )

    fig.update_layout(
        bargap=0.1,
        xaxis_title='Tỉ lệ tăng trưởng (%)',
        yaxis_title='Tần suất xuất hiện',
        template='plotly_white',
        autosize=True,
        margin=dict(l=20, r=40, t=40, b=20),
        height=400,
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(family="Poppins, Roboto, sans-serif", size=16, color="#333333"),
        modebar_bgcolor='rgba(0,0,0,0)',
        modebar_activecolor='#007bff',
        modebar_color='#333333'
    )

    fig.update_xaxes(showgrid=False)
    fig.update_yaxes(showgrid=False)

    return fig

#Biểu đồ phân tán
def plot_close_vs_volume_scatter_with_correlation(df):
    import uuid

    if 'Close' not in df.columns or 'Volume' not in df.columns:
        st.error("❌ Dữ liệu cần có cả cột 'Close' và 'Volume'")
        return

    x = df['Close']
    y = df['Volume']
    correlation = x.corr(y)
    slope_raw = np.cov(x, y)[0, 1] / np.var(x)
    r_squared = correlation ** 2

    def format_slope(value):
        abs_val = abs(value)
        if abs_val >= 1e9:
            return f"{value / 1e9:.2f} B"
        elif abs_val >= 1e6:
            return f"{value / 1e6:.2f} M"
        elif abs_val >= 1e3:
            return f"{value / 1e3:.2f} K"
        else:
            return f"{value:.2f}"

    slope_display = format_slope(slope_raw)

    with st.container():
        st.markdown('<div class="stat-container-native">', unsafe_allow_html=True)
        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown("""
            <div class="stat-metric">
                <h4>Hệ số tương quan (Pearson)</h4>
                <div class="value">{:.4f}</div>
                <div class="description">Mối liên hệ tuyến tính giữa giá và khối lượng.</div>
            </div>
            """.format(correlation), unsafe_allow_html=True)

        with col2:
            st.markdown("""
            <div class="stat-metric">
                <h4>Độ Dốc hồi quy</h4>
                <div class="value">{}</div>
                <div class="description">Volume tăng trung bình khi Close tăng 1 đơn vị.</div>
            </div>
            """.format(slope_display), unsafe_allow_html=True)

        with col3:
            st.markdown("""
            <div class="stat-metric">
                <h4>Hệ số xác định R²</h4>
                <div class="value">{:.4f}</div>
                <div class="description">% biến thiên Volume giải thích bởi Close.</div>
            </div>
            """.format(r_squared), unsafe_allow_html=True)

        st.markdown('</div>', unsafe_allow_html=True)

    # Biểu đồ phân tán
    fig = px.scatter(
        df,
        x='Close',
        y='Volume',
        title='Phân Tán: Giá Đóng Cửa vs Khối Lượng Giao Dịch',
        labels={'Close': 'Giá Đóng Cửa', 'Volume': 'Khối Lượng Giao Dịch'},
        opacity=0.7,
        color='Close',
        color_continuous_scale='Viridis'
    )

    fig.update_traces(marker=dict(size=8, line=dict(width=0.5, color='DarkSlateGrey')))
    fig.update_layout(
        template='plotly_white',
        height=500,
        margin=dict(l=20, r=20, t=60, b=40),
        autosize=True,
        font=dict(family="Poppins, Roboto, sans-serif", size=16, color="#333333")
    )

    st.plotly_chart(fig, use_container_width=True, key=f"scatter_{uuid.uuid4()}")



#Biểu đồ hộp
def plot_close_boxplot(df):

    df = df.copy()
    df['Date'] = pd.to_datetime(df['Date'])

    fig = px.box(
        df,
        y='Close',
        points='all',  
        labels={'Close': 'Giá đóng cửa'},
        color_discrete_sequence=['#007bff']  # Màu chính cho biểu đồ hộp
    )

    fig.update_traces(marker=dict(color='#007bff', size=6),  # Màu cho các điểm outlier
                      line=dict(color='#007bff'))  # Màu viền của box

    fig.update_layout(
        bargap=0.1,
        yaxis_title="Giá đóng cửa",
        xaxis=dict(showticklabels=False),
        template='plotly_white',
        autosize=True,
        margin=dict(l=20, r=40, t=40, b=20),
        height=500,
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(family="Poppins, Roboto, sans-serif", size=16, color="#333333"),
        modebar_bgcolor='rgba(0,0,0,0)',
        modebar_activecolor='#007bff',
        modebar_color='#333333'
    )

    st.plotly_chart(fig, use_container_width=True)


#Chỉ báo kỹ thuật RSI
def plot_rsi(df: pd.DataFrame, window: int = 14):
    import pandas as pd
    import plotly.graph_objects as go

    df = df.copy()
    df['Date'] = pd.to_datetime(df['Date'])
    delta = df['Close'].diff()

    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()

    RS = gain / loss
    RSI = 100 - (100 / (1 + RS))
    df['RSI'] = RSI

    fig = go.Figure()

    # Vẽ đường RSI
    fig.add_trace(go.Scatter(
        x=df['Date'],
        y=df['RSI'],
        mode='lines',
        line=dict(color='#007bff', width=2.5),
        showlegend=False,
        hovertemplate='Ngày: %{x|%d/%m/%Y}<br>RSI: %{y:.2f}<extra></extra>',
        name=''  # Ẩn tên
    ))

    # Highlight vùng bình thường (30–70)
    fig.add_hrect(y0=30, y1=70, fillcolor='rgba(0,123,255,0.10)', line_width=0, layer='below')

    # Vùng quá mua
    fig.add_hrect(y0=70, y1=100, fillcolor='rgba(255,0,0,0.08)', line_width=0, layer='below')

    # Vùng quá bán
    fig.add_hrect(y0=0, y1=30, fillcolor='rgba(34,197,94,0.10)', line_width=0, layer='below')

    # Đường ngưỡng 30 - 70 (dashed)
    fig.add_hline(y=30, line=dict(color="#22c55e", width=1.8, dash="dot"))
    fig.add_hline(y=70, line=dict(color="#ef4444", width=1.8, dash="dot"))

    fig.update_layout(
        height=300,
        template='plotly_white',
        margin=dict(l=30, r=30, t=25, b=30),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(family="Poppins, Roboto, sans-serif", size=15, color="#333333"),
        showlegend=False,
        modebar_bgcolor='rgba(0,0,0,0)',     # Modebar trong suốt
        modebar_activecolor='#007bff',
        modebar_color='#333333',
        xaxis=dict(
            showgrid=False,
            showticklabels=True,
            title='Ngày',
            tickformat='%m/%Y',
            automargin=True
        ),
        yaxis=dict(
            showgrid=False,
            range=[0, 100],
            title='RSI',
            tickvals=[0, 30, 50, 70, 100],
            automargin=True
        )
    )

    st.plotly_chart(fig, use_container_width=True)


#Chỉ báo kỹ thuật log return
def plot_log_return(df: pd.DataFrame):

    df = df.copy()
    df['Date'] = pd.to_datetime(df['Date'])
    if 'log_return' not in df.columns:
        df['log_return'] = np.log(df['Close'] / df['Close'].shift(1))

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df['Date'],
        y=df['log_return'],
        mode='lines',
        line=dict(color='#007bff', width=2),
        hovertemplate='Ngày: %{x|%d/%m/%Y}<br>Log return: %{y:.4f}<extra></extra>',
        name=''
    ))

    fig.update_layout(
        xaxis_title="Ngày",
        yaxis_title="Log Return",
        height=350,
        template='plotly_white',
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(family="Poppins, Roboto, sans-serif", size=15, color="#333333"),
        margin=dict(l=30, r=40, t=30, b=30),
        modebar_bgcolor='rgba(0,0,0,0)',
        modebar_activecolor='#007bff',
        modebar_color='#333333'
    )
    fig.update_xaxes(showgrid=False, tickangle=-45, automargin=True)
    fig.update_yaxes(showgrid=False, automargin=True)

    st.plotly_chart(fig, use_container_width=True)

#Chỉ báo volatility
def plot_price_and_volatility(df: pd.DataFrame):
    import pandas as pd
    import plotly.graph_objects as go

    df = df.copy()
    df['Date'] = pd.to_datetime(df['Date'])
    if 'volatility_5' not in df.columns:
        df['volatility_5'] = df['Close'].rolling(window=5).std()

    fig = go.Figure()

    # Trace 1: Giá đóng cửa
    fig.add_trace(go.Scatter(
        x=df['Date'],
        y=df['Close'],
        mode='lines',
        line=dict(color='#22223b', width=2),
        name='Close',
        yaxis='y1',
        showlegend=False,
        hovertemplate='Ngày: %{x|%d/%m/%Y}<br>Giá đóng cửa: %{y:.2f}<extra></extra>'
    ))

    # Trace 2: Volatility 5 phiên
    fig.add_trace(go.Scatter(
        x=df['Date'],
        y=df['volatility_5'],
        mode='lines',
        line=dict(color='#007bff', width=2, dash='dot'),
        name='Volatility (std 5 phiên)',
        yaxis='y2',
        showlegend=False,
        hovertemplate='Ngày: %{x|%d/%m/%Y}<br>Volatility 5: %{y:.4f}<extra></extra>'
    ))

    fig.update_layout(
        title="Giá đóng cửa & Volatility 5 phiên",
        xaxis=dict(title="Ngày", showgrid=False, tickangle=-45),
        yaxis=dict(title="Giá đóng cửa", showgrid=False, side='left'),
        yaxis2=dict(title="Volatility (std 5 phiên)", showgrid=False, overlaying='y', side='right'),
        height=400,
        template='plotly_white',
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(family="Poppins, Roboto, sans-serif", size=15, color="#333333"),
        margin=dict(l=30, r=30, t=35, b=30),
        showlegend=False,
        modebar_bgcolor='rgba(0,0,0,0)',
        modebar_activecolor='#007bff',
        modebar_color='#333333'
    )

    st.plotly_chart(fig, use_container_width=True)


def plot_momentum_5(df: pd.DataFrame):
    import pandas as pd
    import plotly.graph_objects as go

    df = df.copy()
    df['Date'] = pd.to_datetime(df['Date'])
    if 'momentum_5' not in df.columns:
        df['momentum_5'] = df['Close'] - df['Close'].shift(5)

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df['Date'],
        y=df['momentum_5'],
        mode='lines',
        line=dict(color='#007bff', width=2),
        showlegend=False,
        hovertemplate='Ngày: %{x|%d/%m/%Y}<br>Momentum 5: %{y:.4f}<extra></extra>',
        name=''
    ))

    fig.update_layout(
        title="Biểu đồ Momentum 5 phiên",
        xaxis_title="Ngày",
        yaxis_title="Momentum 5",
        height=350,
        template='plotly_white',
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(family="Poppins, Roboto, sans-serif", size=15, color="#333333"),
        margin=dict(l=30, r=30, t=30, b=30),
        showlegend=False,
        modebar_bgcolor='rgba(0,0,0,0)',
        modebar_activecolor='#007bff',
        modebar_color='#333333'
    )
    fig.update_xaxes(showgrid=False, tickangle=-45, automargin=True)
    fig.update_yaxes(showgrid=False, automargin=True)

    st.plotly_chart(fig, use_container_width=True)


#MACD
def plot_macd(df: pd.DataFrame):
    df = df.copy()
    df['Date'] = pd.to_datetime(df['Date'])

    # Tính MACD nếu chưa có
    if 'macd' not in df.columns or 'macd_signal' not in df.columns:
        ema12 = df['Close'].ewm(span=12, adjust=False).mean()
        ema26 = df['Close'].ewm(span=26, adjust=False).mean()
        df['macd'] = ema12 - ema26
        df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()

    # Histogram (MACD - Signal)
    df['macd_hist'] = df['macd'] - df['macd_signal']

    # Định màu bar: xanh nếu dương, đỏ nếu âm
    bar_colors = np.where(df['macd_hist'] >= 0, '#22c55e', '#ef4444')

    fig = go.Figure()

    # Histogram MACD
    fig.add_trace(go.Bar(
        x=df['Date'],
        y=df['macd_hist'],
        marker_color=bar_colors,
        opacity=0.7,
        showlegend=False,
        hovertemplate='Ngày: %{x|%d/%m/%Y}<br>MACD Hist: %{y:.4f}<extra></extra>',
        name=''
    ))

    # Đường MACD
    fig.add_trace(go.Scatter(
        x=df['Date'],
        y=df['macd'],
        mode='lines',
        line=dict(color='#007bff', width=2),
        showlegend=False,
        hovertemplate='Ngày: %{x|%d/%m/%Y}<br>MACD: %{y:.4f}<extra></extra>',
        name=''
    ))

    # Đường Signal
    fig.add_trace(go.Scatter(
        x=df['Date'],
        y=df['macd_signal'],
        mode='lines',
        line=dict(color='#FF9800', width=2, dash='dot'),
        showlegend=False,
        hovertemplate='Ngày: %{x|%d/%m/%Y}<br>Signal: %{y:.4f}<extra></extra>',
        name=''
    ))

    fig.update_layout(
        title="Biểu đồ MACD",
        xaxis_title="Ngày",
        yaxis_title="MACD",
        height=350,
        template='plotly_white',
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(family="Poppins, Roboto, sans-serif", size=15, color="#333333"),
        margin=dict(l=30, r=30, t=30, b=30),
        showlegend=False,
        modebar_bgcolor='rgba(0,0,0,0)',
        modebar_activecolor='#007bff',
        modebar_color='#333333'
    )
    fig.update_xaxes(showgrid=False, tickangle=-45, automargin=True)
    fig.update_yaxes(showgrid=False, automargin=True)

    st.plotly_chart(fig, use_container_width=True)


#plot_bollinger_bands
def plot_bollinger_bands(df: pd.DataFrame, window: int = 20, num_std: int = 2):
    import pandas as pd
    import plotly.graph_objects as go

    df = df.copy()
    df['Date'] = pd.to_datetime(df['Date'])

    # Tính Bollinger Bands nếu chưa có
    if 'bb_upper' not in df.columns or 'bb_lower' not in df.columns:
        ma = df['Close'].rolling(window=window).mean()
        std = df['Close'].rolling(window=window).std()
        df['bb_upper'] = ma + num_std * std
        df['bb_lower'] = ma - num_std * std
        df['bb_ma'] = ma
    else:
        df['bb_ma'] = df['Close'].rolling(window=window).mean()

    fig = go.Figure()

    # Vùng giữa 2 bands (fill)
    fig.add_trace(go.Scatter(
        x=pd.concat([df['Date'], df['Date'][::-1]]),
        y=pd.concat([df['bb_upper'], df['bb_lower'][::-1]]),
        fill='toself',
        fillcolor='rgba(0,123,255,0.12)',   # Vùng xanh nhạt
        line=dict(color='rgba(255,255,255,0)'),  # Ẩn viền vùng
        hoverinfo='skip',
        showlegend=False,
        name=''
    ))

    # Đường Upper band
    fig.add_trace(go.Scatter(
        x=df['Date'],
        y=df['bb_upper'],
        mode='lines',
        line=dict(color='#007bff', width=1.5, dash='dot'),
        showlegend=False,
        hovertemplate='Upper Band: %{y:.2f}<extra></extra>',
        name=''
    ))

    # Đường Lower band
    fig.add_trace(go.Scatter(
        x=df['Date'],
        y=df['bb_lower'],
        mode='lines',
        line=dict(color='#007bff', width=1.5, dash='dot'),
        showlegend=False,
        hovertemplate='Lower Band: %{y:.2f}<extra></extra>',
        name=''
    ))

    # Đường trung bình (MA20)
    fig.add_trace(go.Scatter(
        x=df['Date'],
        y=df['bb_ma'],
        mode='lines',
        line=dict(color='#ff9800', width=1.8),
        showlegend=False,
        hovertemplate='MA20: %{y:.2f}<extra></extra>',
        name=''
    ))

    # Đường giá Close
    fig.add_trace(go.Scatter(
        x=df['Date'],
        y=df['Close'],
        mode='lines',
        line=dict(color='#22223b', width=2),
        showlegend=False,
        hovertemplate='Ngày: %{x|%d/%m/%Y}<br>Close: %{y:.2f}<extra></extra>',
        name=''
    ))

    fig.update_layout(
        title="Bollinger Bands & Giá đóng cửa",
        xaxis_title="Ngày",
        yaxis_title="Giá đóng cửa",
        height=350,
        template='plotly_white',
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(family="Poppins, Roboto, sans-serif", size=15, color="#333333"),
        margin=dict(l=30, r=30, t=30, b=30),
        showlegend=False,
        modebar_bgcolor='rgba(0,0,0,0)',
        modebar_activecolor='#007bff',
        modebar_color='#333333'
    )
    fig.update_xaxes(showgrid=False, tickangle=-45, automargin=True)
    fig.update_yaxes(showgrid=False, automargin=True)

    st.plotly_chart(fig, use_container_width=True)
















#Biểu đồ giá & khối lượng giao dịch
def plot_price_movement_chart(df, start_date, end_date):

    # Reset index nếu cần
    if 'Date' not in df.columns:
        df = df.reset_index()

    df['Date'] = pd.to_datetime(df['Date']).dt.tz_localize(None)

    # Lọc theo khoảng thời gian
    start_date = pd.to_datetime(start_date)
    end_date = pd.to_datetime(end_date)
    df_filtered = df[(df['Date'] >= start_date) & (df['Date'] <= end_date)].copy()

    # Tạo biểu đồ
    fig = make_subplots(specs=[[{"secondary_y": True}]])

    fig.add_trace(
        go.Scatter(x=df_filtered['Date'], y=df_filtered['Close'], mode='lines', name='Close', line=dict(color='green')),
        secondary_y=False
    )

    fig.add_trace(
        go.Scatter(x=df_filtered['Date'], y=df_filtered['Open'], mode='lines', name='Open', line=dict(color='blue')),
        secondary_y=False
    )

    fig.add_trace(
        go.Bar(x=df_filtered['Date'], y=df_filtered['Volume'], name='Volume', marker_color='rgba(0, 0, 0, 0.6)'),
        secondary_y=True
    )

    # Cập nhật layout với legend xếp hàng ngang, căn trái
    fig.update_layout(
        xaxis_title="Ngày",
        yaxis_title="Giá",
        yaxis2_title="Khối lượng",
        height=600,
        barmode='overlay',
        xaxis_rangeslider_visible=True,
        template='plotly_white',
        autosize=True,
        margin=dict(l=20, r=20, t=40, b=20),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(family="Poppins, Roboto, sans-serif", size=16, color="#333333"),
        legend=dict(
            orientation="h",  # Hàng ngang
            yanchor="bottom",
            y=1.02,           # Đặt phía trên biểu đồ
            xanchor="left",
            x=0,
            title_text=''     # Loại bỏ tiêu đề chú giải nếu có
        ),
        modebar_bgcolor='rgba(0,0,0,0)',
        modebar_activecolor='#007bff',
        modebar_color='#333333'
    )

    # Xóa gridline
    fig.update_xaxes(showgrid=False)
    fig.update_yaxes(showgrid=False, secondary_y=False)
    fig.update_yaxes(showgrid=False, secondary_y=True)

    return fig




#Biểu đồ nến
def plot_candlestick_chart(df, start_date, end_date):

    # Đảm bảo có cột 'Date'
    if 'Date' not in df.columns:
        df = df.reset_index()

    df['Date'] = pd.to_datetime(df['Date']).dt.tz_localize(None)

    # Lọc dữ liệu theo khoảng thời gian
    start_date = pd.to_datetime(start_date)
    end_date = pd.to_datetime(end_date)
    df_filtered = df[(df['Date'] >= start_date) & (df['Date'] <= end_date)].copy()

    # Tạo biểu đồ nến
    fig = go.Figure(data=[
        go.Candlestick(
            x=df_filtered['Date'],
            open=df_filtered['Open'],
            high=df_filtered['High'],
            low=df_filtered['Low'],
            close=df_filtered['Close'],
            name='Biểu đồ Nến'
        )
    ])

    # Thêm chú giải nguyên lý mà không có viền
    fig.add_annotation(
        text=(
            "- Nến tăng: Đóng > Mở (màu xanh).<br>"
            "- Nến giảm: Đóng < Mở (màu đỏ).<br>"
            "- Đuôi nến: Giá cao nhất và thấp nhất trong phiên."
        ),
        align='left',
        showarrow=False,
        xref='paper', yref='paper',
        x=0, y=1.15,  # Vị trí trên cùng biểu đồ
        borderwidth=0,  # Bỏ viền
        bgcolor="rgba(255, 255, 255, 0.8)",  # Nền mờ nhẹ, có thể chỉnh về 0.0 nếu muốn trong suốt hoàn toàn
        font=dict(family="Poppins, Roboto, sans-serif", size=12, color="#333333")
    )

    # Cập nhật layout
    fig.update_layout(
        xaxis_title='Ngày',
        yaxis_title='Giá',
        xaxis_rangeslider_visible=True,
        height=500,
        template='plotly_white',
        autosize=True,
        margin=dict(l=20, r=40, t=100, b=20),  # Đủ chỗ cho phần chú giải
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(family="Poppins, Roboto, sans-serif", size=16, color="#333333"),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="left",
            x=0
        ),
        modebar_bgcolor='rgba(0,0,0,0)',
        modebar_activecolor='#007bff',
        modebar_color='#333333'
    )

    # Xóa gridline
    fig.update_xaxes(showgrid=False)
    fig.update_yaxes(showgrid=False)

    return fig


# Giá trị giao dịch theo tháng
def plot_total_traded_value_by_month(df: pd.DataFrame):
    df = df.copy()
    df['Date'] = pd.to_datetime(df['Date'])
    df['Month'] = df['Date'].dt.month
    df['Year'] = df['Date'].dt.year

    df['Traded Value'] = df['Close'] * df['Volume']

    monthly_total = df.groupby(['Year', 'Month'])['Traded Value'].sum().reset_index()
    monthly_total['Label'] = monthly_total['Month'].astype(str).str.zfill(2) + '/' + monthly_total['Year'].astype(str)

    fig = go.Figure()

    # Vùng (area)
    fig.add_trace(go.Scatter(
        x=monthly_total['Label'],
        y=monthly_total['Traded Value'],
        mode='lines',
        fill='tozeroy',
        fillcolor='rgba(0,123,255,0.35)',    # Vùng xanh nhạt, alpha đồng bộ
        line=dict(color='#007bff', width=2.5),  # Line xanh đồng bộ quý
        name='Tổng giá trị giao dịch'
    ))

    fig.update_layout(
        xaxis_title='Tháng/Năm',
        yaxis_title='Tổng giá trị giao dịch',
        height=500,
        template='plotly_white',
        autosize=True,
        margin=dict(l=20, r=40, t=40, b=40),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(family="Poppins, Roboto, sans-serif", size=16, color="#333333"),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="left",
            x=0
        ),
        modebar_bgcolor='rgba(0,0,0,0)',
        modebar_activecolor='#007bff',
        modebar_color='#333333',
    )
    fig.update_xaxes(showgrid=False, automargin=True, tickangle=-45)
    fig.update_yaxes(showgrid=False, automargin=True)

    st.plotly_chart(fig, use_container_width=True)


# Giá trị giao dịch theo quý   
def plot_total_traded_value_by_quarter(df: pd.DataFrame):
    df = df.copy()
    df['Date'] = pd.to_datetime(df['Date'])
    df['Traded Value'] = df['Close'] * df['Volume']
    df['Quarter'] = df['Date'].dt.to_period('Q').astype(str)
    quarterly_total = df.groupby('Quarter')['Traded Value'].sum().reset_index()

    fig = go.Figure()

    # Area chart (chỉ 1 trace như tháng)
    fig.add_trace(go.Scatter(
        x=quarterly_total['Quarter'],
        y=quarterly_total['Traded Value'],
        mode='lines',
        fill='tozeroy',
        fillcolor='rgba(0,123,255,0.35)',     # Màu vùng đồng bộ tháng
        line=dict(color='#007bff', width=2.5), # Line đồng bộ tháng
        name='Tổng giá trị giao dịch'
    ))

    fig.update_layout(
        xaxis_title='Quý',
        yaxis_title='Tổng giá trị giao dịch',
        height=500,
        template='plotly_white',
        autosize=True,
        margin=dict(l=20, r=40, t=40, b=40),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(family="Poppins, Roboto, sans-serif", size=16, color="#333333"),
        showlegend=False,   # Ẩn chú giải
        modebar_bgcolor='rgba(0,0,0,0)',
        modebar_activecolor='#007bff',
        modebar_color='#333333'
    )
    fig.update_xaxes(showgrid=False, automargin=True, tickangle=-45)
    fig.update_yaxes(showgrid=False, automargin=True)

    st.plotly_chart(fig, use_container_width=True)











# Biểu đồ giá theo thứ trong tuần
def plot_weekday_analysis_chart(df):
    df = df.copy()
    df['Date'] = pd.to_datetime(df['Date'])
    df['Weekday'] = df['Date'].dt.day_name()
    order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday']

    # Tính tăng trưởng
    df['Return'] = df['Close'].pct_change()

    # Trung bình theo thứ
    avg_close = df.groupby('Weekday')['Close'].mean().reindex(order)
    avg_growth = df.groupby('Weekday')['Return'].mean().reindex(order) * 100

    # Biểu đồ kết hợp
    fig = go.Figure()

    fig.add_trace(go.Bar(
        x=avg_close.index,
        y=avg_close.values,
        name='Giá đóng cửa TB',
        marker_color='steelblue',
        yaxis='y1'
    ))

    fig.add_trace(go.Scatter(
        x=avg_growth.index,
        y=avg_growth.values,
        name='Tăng trưởng TB (%)',
        mode='lines+markers',
        line=dict(color='darkorange', width=3),
        yaxis='y2'
    ))

    fig.update_layout(
        xaxis=dict(title="Thứ trong tuần"),
        yaxis=dict(title="Giá đóng cửa TB", side='left'),
        yaxis2=dict(title="Tăng trưởng TB (%)", overlaying='y', side='right', showgrid=False),
        legend=dict(x=0.01, y=0.99),
        bargap=0.3,
        height=500
    )

    st.plotly_chart(fig, use_container_width=True)

# Biểu đồ giá theo tháng trong năm
def plot_combined_chart_by_month(df):
    df = df.copy()
    df['Date'] = pd.to_datetime(df['Date'])
    df['Month'] = df['Date'].dt.month  # Lấy tháng dạng số (1–12)

    # Tính tăng trưởng
    df['Return'] = df['Close'].pct_change()

    # Trung bình giá và tăng trưởng theo tháng
    avg_close = df.groupby('Month')['Close'].mean()
    avg_growth = df.groupby('Month')['Return'].mean() * 100

    month_labels = ['Tháng 1', 'Tháng 2', 'Tháng 3', 'Tháng 4', 'Tháng 5', 'Tháng 6',
                    'Tháng 7', 'Tháng 8', 'Tháng 9', 'Tháng 10', 'Tháng 11', 'Tháng 12']

    # Tạo biểu đồ kết hợp
    fig = go.Figure()

    fig.add_trace(go.Bar(
        x=month_labels,
        y=avg_close.values,
        name='Giá đóng cửa TB',
        marker_color='steelblue',
        yaxis='y1'
    ))

    fig.add_trace(go.Scatter(
        x=month_labels,
        y=avg_growth.values,
        name='Tăng trưởng TB (%)',
        mode='lines+markers',
        line=dict(color='darkorange', width=3),
        yaxis='y2'
    ))

    fig.update_layout(
        xaxis=dict(title="Tháng"),
        yaxis=dict(title="Giá đóng cửa TB", side='left'),
        yaxis2=dict(title="Tăng trưởng TB (%)", overlaying='y', side='right', showgrid=False),
        legend=dict(x=0.01, y=0.99),
        bargap=0.3,
        height=500
    )

    st.plotly_chart(fig, use_container_width=True)



#Biểu đồ volume theo thứ
def plot_volume_and_growth_by_weekday(df: pd.DataFrame):
    df = df.copy()
    df['Date'] = pd.to_datetime(df['Date'])
    df['Weekday'] = df['Date'].dt.day_name()
    weekday_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday']

    # Tính tăng trưởng khối lượng giao dịch (%)
    df['Volume Growth (%)'] = df['Volume'].pct_change() * 100

    # Tính trung bình Volume và Volume Growth theo thứ
    avg_volume = df.groupby('Weekday')['Volume'].mean().reindex(weekday_order)
    avg_volume_growth = df.groupby('Weekday')['Volume Growth (%)'].mean().reindex(weekday_order)

    # Tạo biểu đồ kết hợp
    fig = go.Figure()

    # Volume trung bình (cột)
    fig.add_trace(go.Bar(
        x=avg_volume.index,
        y=avg_volume.values,
        name="Khối lượng giao dịch TB",
        marker_color='teal',
        yaxis='y1'
    ))

    # Tăng trưởng khối lượng trung bình (đường)
    fig.add_trace(go.Scatter(
        x=avg_volume_growth.index,
        y=avg_volume_growth.values,
        name="Tăng trưởng khối lượng TB (%)",
        mode='lines+markers',
        line=dict(color='darkorange', width=3),
        yaxis='y2'
    ))

    fig.update_layout(
        title="Phân tích khối lượng & tăng trưởng khối lượng theo thứ trong tuần",
        xaxis=dict(title="Thứ trong tuần"),
        yaxis=dict(title="Khối lượng giao dịch TB", side='left'),
        yaxis2=dict(title="Tăng trưởng khối lượng TB (%)", overlaying='y', side='right', showgrid=False),
        legend=dict(x=0.01, y=0.99),
        height=500,
        template='plotly_white'
    )

    st.plotly_chart(fig, use_container_width=True)
    
#Biều đồ volume theo tháng
def plot_volume_and_growth_by_month(df: pd.DataFrame):
    df = df.copy()
    df['Date'] = pd.to_datetime(df['Date'])
    df['Month'] = df['Date'].dt.month  # Số tháng: 1-12

    # Tính tăng trưởng khối lượng giao dịch (%)
    df['Volume Growth (%)'] = df['Volume'].pct_change() * 100

    # Trung bình khối lượng và tăng trưởng khối lượng theo tháng
    avg_volume = df.groupby('Month')['Volume'].mean()
    avg_volume_growth = df.groupby('Month')['Volume Growth (%)'].mean()

    # Nhãn tháng tiếng Việt
    month_labels = ['Tháng 1', 'Tháng 2', 'Tháng 3', 'Tháng 4', 'Tháng 5', 'Tháng 6',
                    'Tháng 7', 'Tháng 8', 'Tháng 9', 'Tháng 10', 'Tháng 11', 'Tháng 12']

    # Tạo biểu đồ
    fig = go.Figure()

    # Cột khối lượng giao dịch
    fig.add_trace(go.Bar(
        x=month_labels,
        y=avg_volume.values,
        name='Khối lượng giao dịch TB',
        marker_color='teal',
        yaxis='y1'
    ))

    # Đường tăng trưởng khối lượng
    fig.add_trace(go.Scatter(
        x=month_labels,
        y=avg_volume_growth.values,
        name='Tăng trưởng khối lượng TB (%)',
        mode='lines+markers',
        line=dict(color='darkorange', width=3),
        yaxis='y2'
    ))

    # Layout biểu đồ
    fig.update_layout(
        title='Phân tích khối lượng & tăng trưởng khối lượng theo tháng trong năm',
        xaxis=dict(title="Tháng"),
        yaxis=dict(title="Khối lượng giao dịch TB", side='left'),
        yaxis2=dict(title="Tăng trưởng khối lượng TB (%)", overlaying='y', side='right', showgrid=False),
        legend=dict(x=0.01, y=0.99),
        bargap=0.3,
        height=500,
        template='plotly_white'
    )

    st.plotly_chart(fig, use_container_width=True)



#Tổng giao dịch từng tháng
def plot_total_and_avg_close_combined_by_month(df: pd.DataFrame):
    df = df.copy()
    df['Date'] = pd.to_datetime(df['Date'])
    df['Month'] = df['Date'].dt.to_period('M')

    monthly_stats = df.groupby('Month')['Close'].agg(Total='sum', Average='mean').reset_index()
    monthly_stats['Month'] = monthly_stats['Month'].dt.to_timestamp()

    fig = go.Figure()

    # Bar: Tổng giá đóng cửa
    fig.add_trace(go.Bar(
        x=monthly_stats['Month'],
        y=monthly_stats['Total'],
        name='Tổng giá đóng cửa',
        marker_color='#007bff',
        yaxis='y1'
    ))

    # Line: Trung bình giá đóng cửa
    fig.add_trace(go.Scatter(
        x=monthly_stats['Month'],
        y=monthly_stats['Average'],
        mode='lines',
        name='Trung bình giá đóng cửa',
        line=dict(color='#FF0000', width=3),
        marker=dict(size=6, color='#FF0000', line=dict(width=1, color='white')),
        yaxis='y2'
    ))

    fig.update_layout(
        xaxis=dict(title='Tháng', showgrid=False),  # Bỏ gridline trục X
        yaxis=dict(title='Tổng giá đóng cửa', side='left', showgrid=False),  # Bỏ gridline trục Y
        yaxis2=dict(title='Trung bình giá đóng cửa', overlaying='y', side='right', showgrid=False),
        showlegend=False,  # Ẩn legend
        height=500,
        template='plotly_white',
        bargap=0.3,
        font=dict(family="Poppins, Roboto, sans-serif", size=16, color="#333333"),
        margin=dict(l=20, r=20, t=40, b=40)
    )

    st.plotly_chart(fig, use_container_width=True)




#Biều đồ tự tương quan
def plot_interactive_autocorrelation(df, lags=30):
    df = df.copy()
    df['Date'] = pd.to_datetime(df['Date'])

    # Tính log return để làm chuỗi stationarity
    close_series = np.log(df['Close'] / df['Close'].shift(1)).dropna()

    # Tính autocorrelation
    autocorr_values = [close_series.autocorr(lag=i) for i in range(1, lags + 1)]

    # Vẽ biểu đồ
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=list(range(1, lags + 1)),
        y=autocorr_values,
        marker_color='#1E90FF',
        name='Tự tương quan'
    ))

    fig.update_layout(
        title="Biểu đồ Tự Tương Quan (ACF) – Log Return",
        xaxis_title="Độ trễ (Lag)",
        yaxis_title="Hệ số tương quan",
        height=500,
        template="plotly_white",
        xaxis=dict(showgrid=False),
        yaxis=dict(showgrid=False)
    )

    st.plotly_chart(fig, use_container_width=True)



#Phân tích chuỗi thời gian
def plot_interactive_decomposition(df):
    df = df.copy()
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.set_index('Date')
    df = df.asfreq('D')  # tùy thuộc dữ liệu, có thể dùng 'B' cho ngày làm việc

    df['Close'] = df['Close'].interpolate()

    decomposition = seasonal_decompose(df['Close'], model='multiplicative', period=30)

    # Tạo figure
    fig = make_subplots(rows=4, cols=1, shared_xaxes=True,
                        subplot_titles=["Dữ liệu gốc", "Xu hướng", "Chu kỳ", "Sai số"])

    fig.add_trace(go.Scatter(x=decomposition.observed.index, y=decomposition.observed,
                             mode='lines', name='Dữ liệu gốc', line=dict(color='green')), row=1, col=1)

    fig.add_trace(go.Scatter(x=decomposition.trend.index, y=decomposition.trend,
                             mode='lines', name='Xu hướng', line=dict(color='blue')), row=2, col=1)

    fig.add_trace(go.Scatter(x=decomposition.seasonal.index, y=decomposition.seasonal,
                             mode='lines', name='Chu kỳ', line=dict(color='orange')), row=3, col=1)

    fig.add_trace(go.Scatter(x=decomposition.resid.index, y=decomposition.resid,
                             mode='lines', name='Sai số', line=dict(color='red')), row=4, col=1)

    fig.update_layout(height=900, showlegend=False)

    st.plotly_chart(fig, use_container_width=True)






def create_adj_close_multi_ma_chart_with_prediction(
    df_train,
    ma_windows=[20],
    symbol='A',
    test_folder='D:/Data Science/yfinance/',
    forecast_days=5
):
    import os
    import pandas as pd
    import numpy as np
    from sklearn.metrics import mean_absolute_error, mean_squared_error
    import plotly.graph_objs as go

    # Đọc file test dựa trên symbol
    test_path = os.path.join(test_folder, f"{symbol}.csv")
    df_test = pd.read_csv(test_path)
    if 'Date' in df_test.columns:
        df_test['Date'] = pd.to_datetime(df_test['Date'])
        df_test = df_test.set_index('Date')

    # Chỉ lấy đúng số ngày dự đoán
    df_test = df_test.iloc[:forecast_days]

    # Xử lý index cho train
    df_train = df_train.copy()
    if 'Date' in df_train.columns:
        df_train['Date'] = pd.to_datetime(df_train['Date'])
        df_train = df_train.set_index('Date')

    # Tính MA cho train
    for window in ma_windows:
        df_train[f"MA{window}"] = df_train['Close'].rolling(window=window).mean()

    prediction_table = {}
    error_table = []

    # Dự báo từng MA
    for window in ma_windows:
        # Lấy chuỗi cuối cho dự báo
        last_close = df_train['Close'].tail(window).values
        preds = []
        for i in range(forecast_days):
            if i == 0:
                pred = np.mean(last_close)
            else:
                pred = np.mean(np.concatenate([last_close[i:], preds]))
            preds.append(pred)
        prediction_table[window] = preds

        # Tính lỗi Train (fitted)
        fitted = df_train[f"MA{window}"].dropna()
        actual_train = df_train['Close'].iloc[window - 1:]
        mae_train = mean_absolute_error(actual_train, fitted)
        rmse_train = np.sqrt(mean_squared_error(actual_train, fitted))
        mape_train = np.mean(np.abs((actual_train - fitted) / actual_train)) * 100

        # Lỗi future (predict)
        actual_future = df_test['Close'].values
        if len(actual_future) == len(preds):
            mae_pred = mean_absolute_error(actual_future, preds)
            rmse_pred = np.sqrt(mean_squared_error(actual_future, preds))
            mape_pred = np.mean(np.abs((actual_future - preds) / actual_future)) * 100
        else:
            mae_pred = rmse_pred = mape_pred = None

        error_table.append([
            f"MA{window}",
            mae_train, rmse_train, mape_train,
            None, None, None,
            mae_pred, rmse_pred, mape_pred
        ])

    # Tạo DataFrame kết quả dự đoán
    future_dates = df_test.index[:forecast_days]
    df_pred = pd.DataFrame(
        {f"Prediction_MA{w}": prediction_table[w] for w in ma_windows},
        index=future_dates
    )
    df_pred["Actual Close"] = df_test['Close'].values

    # Tạo bảng lỗi
    error_df = pd.DataFrame(error_table, columns=[
        "MA",
        "Train MAE", "Train RMSE", "Train MAPE (%)",
        "Test MAE", "Test RMSE", "Test MAPE (%)",
        "Predict MAE", "Predict RMSE", "Predict MAPE (%)"
    ])

    # -------- Vẽ biểu đồ --------
    fig = go.Figure()

    # Đường Close (train)
    fig.add_trace(go.Scatter(
        x=df_train.index, y=df_train['Close'],
        mode='lines', name='Close (Train)',
        line=dict(color='#1E90FF', width=3)
    ))

    # Đường Close thực tế (test/future)
    fig.add_trace(go.Scatter(
        x=df_test.index, y=df_test['Close'],
        mode='lines', name='Close (Thực tế)',
        line=dict(color='black', width=3)
    ))

    # Các đường MA (train) + dự báo MA
    palette = ['#ff9800', '#009688', '#e91e63', '#9c27b0', '#00bcd4', '#4caf50', '#f44336']
    for idx, window in enumerate(ma_windows):
        color = palette[idx % len(palette)]
        fig.add_trace(go.Scatter(
            x=df_train.index, y=df_train[f"MA{window}"],
            mode='lines', name=f"Fitted MA {window}",
            line=dict(color=color, width=2, dash='dot')
        ))
        fig.add_trace(go.Scatter(
            x=future_dates, y=prediction_table[window],
            mode='lines', name=f'Predict MA {window}',
            line=dict(color=color, width=2, dash='dash')
        ))

    fig.update_layout(
        title_text=f"Biểu đồ Close, Multi-MA, Dự báo & Giá thực tế ({symbol})",
        xaxis_title="Ngày",
        yaxis_title="Giá trị",
        height=600,
        plot_bgcolor='white',
        xaxis=dict(showgrid=False),
        yaxis=dict(showgrid=False),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=40, r=40, t=80, b=40)
    )

    return fig, df_pred, error_df


def calc_ma_prediction_with_real_test(
    df,
    ma_windows=[20],
    forecast_days=5,
    train_ratio=0.8,
    test_folder='D:/Data Science/yfinance/',
    symbol='A'
):
    import pandas as pd
    import numpy as np
    from sklearn.metrics import mean_absolute_error, mean_squared_error
    import os

    df = df.copy()
    if 'Date' not in df.columns:
        df = df.reset_index()
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values('Date').reset_index(drop=True)

    n = len(df)
    n_train = int(n * train_ratio)
    df_train = df.iloc[:n_train]
    df_test = df.iloc[n_train:]

    # Đọc dữ liệu thực tế từ thư mục yfinance
    future_path = os.path.join(test_folder, f"{symbol}.csv")
    df_future = pd.read_csv(future_path)
    df_future['Date'] = pd.to_datetime(df_future['Date'])
    df_future = df_future.sort_values('Date').reset_index(drop=True)
    df_future = df_future.iloc[:forecast_days]

    prediction_tables = {}
    error_rows = []

    for window in ma_windows:
        # Train MA
        df_train[f"MA{window}"] = df_train['Close'].rolling(window).mean()
        train_ma = df_train[f"MA{window}"].dropna()
        train_actual = df_train['Close'].loc[train_ma.index]
        train_dates = df_train['Date'].loc[train_ma.index]

        # Test (walk-forward)
        test_preds = []
        history = df_train['Close'].copy()
        for i in range(len(df_test)):
            pred = history.tail(window).mean()
            test_preds.append(pred)
            history = pd.concat([history, pd.Series(df_test['Close'].iloc[i])], ignore_index=True)

        # Future forecast (recursive)
        future_preds = []
        history2 = pd.concat([df_train['Close'], df_test['Close']], ignore_index=True)
        for i in range(forecast_days):
            pred = history2.tail(window).mean()
            future_preds.append(pred)
            history2 = pd.concat([history2, pd.Series(pred)], ignore_index=True)

        # Lưu dữ liệu dự báo
        prediction_tables[window] = {
            "Train Dates": train_dates,
            "Train Actual": train_actual,
            "Train MA": train_ma,
            "Test Dates": df_test['Date'],
            "Test Actual": df_test['Close'],
            "Test Predict": test_preds,
            "Future Dates": df_future['Date'],
            "Future Predict": future_preds,
            "Future Actual": df_future['Close']
        }

        # Tính lỗi
        mae_train = mean_absolute_error(train_actual, train_ma)
        rmse_train = np.sqrt(mean_squared_error(train_actual, train_ma))
        mape_train = np.mean(np.abs((train_actual - train_ma) / train_actual)) * 100

        mae_test = mean_absolute_error(df_test['Close'], test_preds)
        rmse_test = np.sqrt(mean_squared_error(df_test['Close'], test_preds))
        mape_test = np.mean(np.abs((df_test['Close'] - test_preds) / df_test['Close'])) * 100

        mae_future = mean_absolute_error(df_future['Close'], future_preds)
        rmse_future = np.sqrt(mean_squared_error(df_future['Close'], future_preds))
        mape_future = np.mean(np.abs((df_future['Close'] - future_preds) / df_future['Close'])) * 100

        error_rows.append([
            window, mae_train, rmse_train, mape_train,
            mae_test, rmse_test, mape_test,
            mae_future, rmse_future, mape_future
        ])

    error_df = pd.DataFrame(error_rows, columns=[
        'MA', 'MAE_Train', 'RMSE_Train', 'MAPE_Train (%)',
        'MAE_Test', 'RMSE_Test', 'MAPE_Test (%)',
        'MAE_Future', 'RMSE_Future', 'MAPE_Future (%)'
    ])

    return prediction_tables, error_df






#ES
def create_adj_close_es_chart_with_prediction(
    df, smoothing_level=0.1, train_ratio=0.7, test_ratio=0.2, forecast_days=10,
    symbol='A', test_folder='D:/Data Science/yfinance/'
):

    # Chuẩn hóa dữ liệu
    if 'Date' not in df.columns:
        df = df.reset_index()
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values('Date').reset_index(drop=True)

    n = len(df)
    n_train = int(n * train_ratio)
    n_test = int(n * test_ratio)

    df_train = df.iloc[:n_train]
    df_test = df.iloc[n_train:n_train + n_test]

    # Load dữ liệu thực tế tương lai nếu có
    fpath = os.path.join(test_folder, f"{symbol}.csv")
    if os.path.exists(fpath):
        df_yf = pd.read_csv(fpath)
        df_yf['Date'] = pd.to_datetime(df_yf['Date'])
        df_yf = df_yf.sort_values('Date').reset_index(drop=True)
        future_days = df_yf['Date'].iloc[:forecast_days].tolist()
        actual_future = df_yf['Close'].iloc[:forecast_days].tolist()
    else:
        last_date = df['Date'].iloc[-1]
        future_days = [last_date + pd.Timedelta(days=i+1) for i in range(forecast_days)]
        actual_future = [None] * forecast_days

    # Fit model train
    model_train = SimpleExpSmoothing(df_train['Close']).fit(
        smoothing_level=smoothing_level, optimized=False
    )
    train_pred = model_train.fittedvalues

    # Dự báo test
    model_test = SimpleExpSmoothing(df_train['Close']).fit(
        smoothing_level=smoothing_level, optimized=False
    )
    test_preds = model_test.forecast(len(df_test))

    # Dự báo future
    model_future = SimpleExpSmoothing(pd.concat([df_train['Close'], df_test['Close']], ignore_index=True)).fit(
        smoothing_level=smoothing_level, optimized=False
    )
    future_preds = model_future.forecast(forecast_days)

    # Tính lỗi
    mae_train = mean_absolute_error(df_train['Close'].iloc[1:], train_pred.iloc[1:])
    rmse_train = np.sqrt(mean_squared_error(df_train['Close'].iloc[1:], train_pred.iloc[1:]))
    mape_train = np.mean(np.abs((df_train['Close'].iloc[1:] - train_pred.iloc[1:]) / df_train['Close'].iloc[1:])) * 100

    mae_test = mean_absolute_error(df_test['Close'], test_preds)
    rmse_test = np.sqrt(mean_squared_error(df_test['Close'], test_preds))
    mape_test = np.mean(np.abs((df_test['Close'] - test_preds) / df_test['Close'])) * 100

    if all(x is not None for x in actual_future):
        mae_future = mean_absolute_error(actual_future, future_preds)
        rmse_future = np.sqrt(mean_squared_error(actual_future, future_preds))
        mape_future = np.mean(np.abs((np.array(actual_future) - np.array(future_preds)) / np.array(actual_future))) * 100
    else:
        mae_future = rmse_future = mape_future = None

    error_df = pd.DataFrame({
        "MAE": [mae_train, mae_test, mae_future],
        "RMSE": [rmse_train, rmse_test, rmse_future],
        "MAPE (%)": [mape_train, mape_test, mape_future]
    }, index=["Train", "Test", "Future"])

    # Bảng dự đoán
    test_pred_df = pd.DataFrame({
        "Ngày": df_test['Date'].values,
        "ES Prediction": test_preds.values,
        "Actual": df_test['Close'].values
    })

    future_pred_df = pd.DataFrame({
        "Ngày": future_days,
        "ES Prediction": future_preds.values,
        "Actual": actual_future,
    })

    # Vẽ biểu đồ
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df_train['Date'], y=df_train['Close'], name='Train Actual', line=dict(color='#1E90FF')))
    fig.add_trace(go.Scatter(x=df_train['Date'], y=train_pred, name='ES Train Fitted', line=dict(color='orange', dash='dot')))
    fig.add_trace(go.Scatter(x=df_test['Date'], y=df_test['Close'], name='Test Actual', line=dict(color='black')))
    fig.add_trace(go.Scatter(x=df_test['Date'], y=test_preds, name='ES Test Prediction', line=dict(color='red', dash='dash')))
    fig.add_trace(go.Scatter(x=future_days, y=future_preds, name='ES Future Forecast', line=dict(color='purple', dash='dot')))
    if all(x is not None for x in actual_future):
        fig.add_trace(go.Scatter(x=future_days, y=actual_future, name='Future Actual', line=dict(color='green')))
    fig.update_layout(
        title=f"Biểu đồ Exponential Smoothing ({symbol}) - Train/Test/Future",
        xaxis_title="Ngày",
        yaxis_title="Giá trị",
        plot_bgcolor='white',
        xaxis=dict(showgrid=False),
        yaxis=dict(showgrid=False),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=40, r=40, t=80, b=40)
    )

    return fig, test_pred_df, future_pred_df, error_df




    

def create_adj_close_holt_chart_with_prediction(
    df, smoothing_level=0.1, smoothing_slope=0.1,
    train_ratio=0.8, test_ratio=0.2, forecast_days=10,
    symbol='A', test_folder='D:/Data Science/yfinance/'
):

    # Xử lý chuẩn hóa dữ liệu
    if 'Date' not in df.columns:
        df = df.reset_index()
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values('Date').reset_index(drop=True)

    # Chia train/test
    n = len(df)
    n_train = int(n * train_ratio)
    n_test = int(n * test_ratio)

    df_train = df.iloc[:n_train]
    df_test = df.iloc[n_train:n_train + n_test]

    # Lấy future từ file nếu có
    fpath = os.path.join(test_folder, f"{symbol}.csv")
    if os.path.exists(fpath):
        df_yf = pd.read_csv(fpath)
        df_yf['Date'] = pd.to_datetime(df_yf['Date'])
        df_yf = df_yf.sort_values('Date').reset_index(drop=True)
        future_days = df_yf['Date'].iloc[:forecast_days].tolist()
        actual_future = df_yf['Close'].iloc[:forecast_days].tolist()
    else:
        last_date = df['Date'].iloc[-1]
        future_days = [last_date + pd.Timedelta(days=i+1) for i in range(forecast_days)]
        actual_future = [None] * forecast_days

    # Fit model train
    model_train = Holt(df_train['Close']).fit(
        smoothing_level=smoothing_level,
        smoothing_trend=smoothing_slope,
        optimized=False
    )
    train_pred = model_train.fittedvalues

    # Dự báo test
    model_test = Holt(df_train['Close']).fit(
        smoothing_level=smoothing_level,
        smoothing_trend=smoothing_slope,
        optimized=False
    )
    test_preds = model_test.forecast(len(df_test))

    # Dự báo future
    model_future = Holt(pd.concat([df_train['Close'], df_test['Close']], ignore_index=True)).fit(
        smoothing_level=smoothing_level,
        smoothing_trend=smoothing_slope,
        optimized=False
    )
    future_preds = model_future.forecast(forecast_days)

    # Tính lỗi
    mae_train = mean_absolute_error(df_train['Close'].iloc[1:], train_pred.iloc[1:])
    rmse_train = np.sqrt(mean_squared_error(df_train['Close'].iloc[1:], train_pred.iloc[1:]))
    mape_train = np.mean(np.abs((df_train['Close'].iloc[1:] - train_pred.iloc[1:]) / df_train['Close'].iloc[1:])) * 100

    mae_test = mean_absolute_error(df_test['Close'], test_preds)
    rmse_test = np.sqrt(mean_squared_error(df_test['Close'], test_preds))
    mape_test = np.mean(np.abs((df_test['Close'] - test_preds) / df_test['Close'])) * 100

    if all(x is not None for x in actual_future):
        mae_future = mean_absolute_error(actual_future, future_preds)
        rmse_future = np.sqrt(mean_squared_error(actual_future, future_preds))
        mape_future = np.mean(np.abs((np.array(actual_future) - np.array(future_preds)) / np.array(actual_future))) * 100
    else:
        mae_future = rmse_future = mape_future = None

    error_df = pd.DataFrame({
        "MAE": [mae_train, mae_test, mae_future],
        "RMSE": [rmse_train, rmse_test, rmse_future],
        "MAPE (%)": [mape_train, mape_test, mape_future]
    }, index=["Train", "Test", "Future"])

    # Bảng kết quả test và future
    test_pred_df = pd.DataFrame({
        "Ngày": df_test['Date'].values,
        "Holt Prediction": test_preds.values,
        "Actual": df_test['Close'].values
    })

    future_pred_df = pd.DataFrame({
        "Ngày": future_days,
        "Holt Prediction": future_preds.values,
        "Actual": actual_future,
    })

    # Biểu đồ
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df_train['Date'], y=df_train['Close'], name='Train Actual', line=dict(color='#1E90FF')))
    fig.add_trace(go.Scatter(x=df_train['Date'], y=train_pred, name='Holt Train Fitted', line=dict(color='orange', dash='dot')))
    fig.add_trace(go.Scatter(x=df_test['Date'], y=df_test['Close'], name='Test Actual', line=dict(color='black')))
    fig.add_trace(go.Scatter(x=df_test['Date'], y=test_preds, name='Holt Test Prediction', line=dict(color='red', dash='dash')))
    fig.add_trace(go.Scatter(x=future_days, y=future_preds, name='Holt Future Forecast', line=dict(color='purple', dash='dot')))
    if all(x is not None for x in actual_future):
        fig.add_trace(go.Scatter(x=future_days, y=actual_future, name='Future Actual', line=dict(color='green')))
    fig.update_layout(
        title=f"Biểu đồ Holt (trend only) - {symbol}",
        xaxis_title="Ngày",
        yaxis_title="Giá trị",
        plot_bgcolor='white',
        xaxis=dict(showgrid=False),
        yaxis=dict(showgrid=False),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=40, r=40, t=80, b=40)
    )


    return fig, test_pred_df, future_pred_df, error_df


def apply_es_monthly(df, smoothing_level, forecast_days):
    """Applies Exponential Smoothing method with monthly aggregation and returns predictions."""

    # Aggregate 'Close' by month
    df['Month'] = df.index.to_period('M')
    monthly_df = df.groupby('Month')['Close'].sum().reset_index()
    monthly_df['Month'] = monthly_df['Month'].dt.to_timestamp()
    monthly_df.set_index('Month', inplace=True)
    monthly_df = monthly_df.dropna(subset=['Close'])
    # Train Exponential Smoothing model on monthly data
    model_es = SimpleExpSmoothing(monthly_df['Close'], initialization_method="estimated").fit(
        smoothing_level=smoothing_level
    )

    # Add historical predictions to DataFrame
    monthly_df['Close ES'] = model_es.fittedvalues

    # Get the last level value
    level = model_es.level[-1]

    # Generate predictions for the next forecast_days
    last_es_value = monthly_df['Close ES'].iloc[-1]  # Lấy giá trị ES cuối cùng

    predictions = []
    for i in range(forecast_days):
        # Dự đoán bằng cách sử dụng giá trị ES cuối cùng (hoặc dự đoán trước đó)
        # và áp dụng smoothing_level
        if i == 0:
            prediction = last_es_value  # Giá trị ban đầu cho dự đoán là giá trị ES cuối cùng
        else:
            prediction = predictions[-1]  # Giá trị dự đoán tiếp theo bằng giá trị dự đoán trước đó
                                        # (vì Simple ES giả định không có xu hướng hoặc tính thời vụ)

        predictions.append(prediction)

    # Create DataFrame for predictions
    future_dates = pd.date_range(start=monthly_df.index[-1] + pd.DateOffset(months=1), periods=forecast_days, freq='M')
    df_pred = pd.DataFrame({'Close ES Prediction': predictions}, index=future_dates)

    # Dự đoán trong khoảng thời gian đó
    predicted_values = predictions  # predictions đã được tính toán trong hàm
    # Get the actual values for the forecast horizon
    actual_values = monthly_df['Close'][-forecast_days:].values

    # Calculate errors using actual and predicted values
    mae = mean_absolute_error(actual_values, predicted_values)  # Replace predicted_values with predictions
    rmse = np.sqrt(mean_squared_error(actual_values, predicted_values))  # Replace predicted_values with predictions
    mape = np.mean(np.abs((actual_values - predicted_values) / actual_values)) * 100  # Replace predicted_values with predictions


    # Generate predictions for the next forecast_days
    predictions = [level] * forecast_days  # ES predictions are constant

    # Create DataFrame for predictions
    future_dates = pd.date_range(start=monthly_df.index[-1] + pd.DateOffset(months=1), periods=forecast_days, freq='M')
    df_pred = pd.DataFrame({'Close ES Prediction': predictions}, index=future_dates)

    # Create the plot
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=monthly_df.index, y=monthly_df['Close'], mode='lines', name='Close (Historical)', line=dict(color='green')))
    fig.add_trace(go.Scatter(x=monthly_df.index, y=monthly_df['Close ES'], mode='lines', name='ES (Historical)', line=dict(color='orange')))
    fig.add_trace(go.Scatter(x=df_pred.index, y=df_pred['Close ES Prediction'], mode='lines', name='ES Prediction', line=dict(color='blue', dash='dash')))
    

    # Update layout
    fig.update_layout(
        title_text="Close and Exponential Smoothing Chart (Monthly Aggregation)",
        xaxis_title="Date",
        yaxis_title="Price",
        xaxis_range=[monthly_df.index.min(), df_pred.index.max()],  # Extend x-axis range to include predictions
        height=600,
    )

    return fig, df_pred, mae, rmse, mape


def create_adj_close_holt_winters_chart_with_prediction(
    df, alpha=0.8, beta=0.2, gamma=0.2, seasonality_periods=7, 
    train_ratio=0.8, test_ratio=0.2, forecast_days=10,
    symbol='A', test_folder='D:/Data Science/yfinance/'
):

    df = df.copy()
    if 'Date' not in df.columns:
        df = df.reset_index()
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values('Date').reset_index(drop=True)

    # Chia tập train/test
    n = len(df)
    n_train = int(n * train_ratio)
    n_test = int(n * test_ratio)

    df_train = df.iloc[:n_train]
    df_test = df.iloc[n_train:n_train + n_test]

    # Load dữ liệu future nếu có
    fpath = os.path.join(test_folder, f"{symbol}.csv")
    if os.path.exists(fpath):
        df_yf = pd.read_csv(fpath)
        df_yf['Date'] = pd.to_datetime(df_yf['Date'])
        df_yf = df_yf.sort_values('Date').reset_index(drop=True)
        future_days = df_yf['Date'].iloc[:forecast_days].tolist()
        actual_future = df_yf['Close'].iloc[:forecast_days].tolist()
    else:
        last_date = df['Date'].iloc[-1]
        future_days = [last_date + pd.Timedelta(days=i + 1) for i in range(forecast_days)]
        actual_future = [None] * forecast_days

    # Fit model cho train
    model_train = ExponentialSmoothing(
        df_train['Close'], trend='add', seasonal='add', seasonal_periods=seasonality_periods
    ).fit(smoothing_level=alpha, smoothing_trend=beta, smoothing_seasonal=gamma, optimized=False)
    train_pred = model_train.fittedvalues.reindex(df_train.index)

    # Dự báo test (multi-step)
    model_test = ExponentialSmoothing(
        df_train['Close'], trend='add', seasonal='add', seasonal_periods=seasonality_periods
    ).fit(smoothing_level=alpha, smoothing_trend=beta, smoothing_seasonal=gamma, optimized=False)
    test_preds = model_test.forecast(len(df_test))

    # Dự báo future
    model_future = ExponentialSmoothing(
        pd.concat([df_train['Close'], df_test['Close']], ignore_index=True),
        trend='add', seasonal='add', seasonal_periods=seasonality_periods
    ).fit(smoothing_level=alpha, smoothing_trend=beta, smoothing_seasonal=gamma, optimized=False)
    future_preds = model_future.forecast(forecast_days)

    # Tính lỗi
    mae_train = mean_absolute_error(df_train['Close'].iloc[1:], train_pred.iloc[1:])
    rmse_train = np.sqrt(mean_squared_error(df_train['Close'].iloc[1:], train_pred.iloc[1:]))
    mape_train = np.mean(np.abs((df_train['Close'].iloc[1:] - train_pred.iloc[1:]) / df_train['Close'].iloc[1:])) * 100

    mae_test = mean_absolute_error(df_test['Close'], test_preds)
    rmse_test = np.sqrt(mean_squared_error(df_test['Close'], test_preds))
    mape_test = np.mean(np.abs((df_test['Close'] - test_preds) / df_test['Close'])) * 100

    if all(x is not None for x in actual_future):
        mae_future = mean_absolute_error(actual_future, future_preds)
        rmse_future = np.sqrt(mean_squared_error(actual_future, future_preds))
        mape_future = np.mean(np.abs((np.array(actual_future) - np.array(future_preds)) / np.array(actual_future))) * 100
    else:
        mae_future = rmse_future = mape_future = None

    error_df = pd.DataFrame({
        "MAE": [mae_train, mae_test, mae_future],
        "RMSE": [rmse_train, rmse_test, rmse_future],
        "MAPE (%)": [mape_train, mape_test, mape_future]
    }, index=["Train", "Test", "Future"])

    test_pred_df = pd.DataFrame({
        "Ngày": df_test['Date'].values,
        "Holt-Winters Prediction": test_preds.values,
        "Actual": df_test['Close'].values
    })

    future_pred_df = pd.DataFrame({
        "Ngày": future_days,
        "Holt-Winters Prediction": future_preds.values,
        "Actual": actual_future,
    })

    # Vẽ biểu đồ
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df_train['Date'], y=df_train['Close'], name='Train Actual', line=dict(color='#1E90FF')))
    fig.add_trace(go.Scatter(x=df_train['Date'], y=train_pred, name='HW Train Fitted', line=dict(color='orange', dash='dot')))
    fig.add_trace(go.Scatter(x=df_test['Date'], y=df_test['Close'], name='Test Actual', line=dict(color='black')))
    fig.add_trace(go.Scatter(x=df_test['Date'], y=test_preds, name='HW Test Prediction', line=dict(color='red', dash='dash')))
    fig.add_trace(go.Scatter(x=future_days, y=future_preds, name='HW Future Forecast', line=dict(color='purple', dash='dot')))
    if all(x is not None for x in actual_future):
        fig.add_trace(go.Scatter(x=future_days, y=actual_future, name='Future Actual', line=dict(color='green')))
    fig.update_layout(
        title=f"Biểu đồ Holt-Winters ({symbol}) - Train/Test/Future",
        xaxis_title="Ngày",
        yaxis_title="Giá trị",
        plot_bgcolor='white',
        xaxis=dict(showgrid=False),
        yaxis=dict(showgrid=False),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=40, r=40, t=80, b=40)
    )

    return fig, test_pred_df, future_pred_df, error_df


def apply_holt_monthly(df, smoothing_level, smoothing_trend, forecast_days):
    """Applies Holt method with monthly aggregation and returns predictions."""

    # Ensure 'Date' column is datetime and set as index
    if 'Date' in df.columns:
        df['Date'] = pd.to_datetime(df['Date']).dt.tz_localize(None)

    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index)

    # Aggregate 'Close' by month
    df['Month'] = df.index.to_period('M')
    monthly_df = df.groupby('Month')['Close'].sum().reset_index()
    monthly_df['Month'] = monthly_df['Month'].dt.to_timestamp()
    monthly_df.set_index('Month', inplace=True)

    # Train Holt model on monthly data
    model_holt = Holt(monthly_df['Close'], initialization_method="estimated").fit(
        smoothing_level=smoothing_level, smoothing_trend=smoothing_trend
    )

    # Add historical predictions to DataFrame
    monthly_df['Close Holt'] = model_holt.fittedvalues

    # Get the last level and trend values
    level = model_holt.level[-1]
    trend = model_holt.trend[-1]

    # Generate predictions for the next forecast_days
    predictions = []
    for i in range(forecast_days):
        prediction = level + trend * (i + 1)  # Holt prediction formula
        predictions.append(prediction)

    # Create DataFrame for predictions
    future_dates = pd.date_range(start=monthly_df.index[-1] + pd.DateOffset(months=1), periods=forecast_days, freq='M')
    df_pred = pd.DataFrame({'Close Holt Prediction': predictions}, index=future_dates)

    # Calculate in-sample errors
    mae = mean_absolute_error(monthly_df['Close'], model_holt.fittedvalues)
    rmse = np.sqrt(mean_squared_error(monthly_df['Close'], model_holt.fittedvalues))
    mape = np.mean(np.abs((monthly_df['Close'] - model_holt.fittedvalues) / monthly_df['Close'])) * 100

    # Create the plot
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=monthly_df.index, y=monthly_df['Close'], mode='lines', name='Close (Historical)', line=dict(color='green')))
    fig.add_trace(go.Scatter(x=monthly_df.index, y=monthly_df['Close Holt'], mode='lines', name='Holt (Historical)', line=dict(color='orange')))
    fig.add_trace(go.Scatter(x=df_pred.index, y=df_pred['Close Holt Prediction'], mode='lines', name='Holt Prediction', line=dict(color='blue', dash='dash')))
    # Update layout
    fig.update_layout(
        title_text="Close and Holt-Winters Chart (Monthly Aggregation)",
        xaxis_title="Date",
        yaxis_title="Price",
        xaxis_range=[monthly_df.index.min(), df_pred.index.max()],  # Extend x-axis range to include predictions
        height=600,
    )

    # Display parameters and error metrics
    st.write(f"Alpha: {smoothing_level:.2f}, Beta: {smoothing_trend:.2f}%")
    st.write(f"**Chỉ số lỗi (Holt):**")
    st.write(f"  - MAE: {mae:.2f}")
    st.write(f"  - RMSE: {rmse:.2f}")
    st.write(f"  - MAPE: {mape:.2f}%")

    return fig, df_pred


def apply_holt_winters_monthly(df, smoothing_level, smoothing_trend, smoothing_seasonal, forecast_days):
    """Applies Holt-Winters method with monthly aggregation and returns predictions."""

    # Ensure 'Date' column is datetime and set as index
    if 'Date' in df.columns:
        df['Date'] = pd.to_datetime(df['Date']).dt.tz_localize(None)

    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index)

    # Aggregate 'Close' by month
    df['Month'] = df.index.to_period('M')
    monthly_df = df.groupby('Month')['Close'].sum().reset_index()
    monthly_df['Month'] = monthly_df['Month'].dt.to_timestamp()
    monthly_df.set_index('Month', inplace=True)

    # Train Holt-Winters model on monthly data
    seasonality_periods = 12  # Set to 12 for yearly seasonality with monthly data
    model_hw = ExponentialSmoothing(
        monthly_df['Close'],
        trend="add",
        seasonal="add",
        seasonal_periods=seasonality_periods,
        initialization_method="estimated"
    ).fit(
        smoothing_level=smoothing_level,
        smoothing_trend=smoothing_trend,
        smoothing_seasonal=smoothing_seasonal
    )

    # Generate future dates for predictions
    future_dates = pd.date_range(start=monthly_df.index[-1] + pd.DateOffset(months=1), periods=forecast_days, freq='MS')  # 'MS' for month start frequency

    # Make predictions
    predictions = model_hw.forecast(forecast_days)

    # Get last values of level and trend
    level = model_hw.level[-1]
    trend = model_hw.trend[-1]

    # Holt-Winters seasonal values are not directly accessible, so we must calculate them
    seasonal_values = model_hw.fittedvalues - (level + trend)

    # Initialize predictions as a list (This is correct)
    predictions = []  

    # Add this line to create the 'Close Holt-Winters' column
    monthly_df['Close Holt-Winters'] = model_hw.fittedvalues  


    # Generate predictions for the next forecast_days
    for i in range(forecast_days):
        seasonal_index = (i + len(df)) % seasonality_periods  # Wrap around seasonality
        prediction = level + trend * (i + 1) + seasonal_values[seasonal_index]
        predictions.append(prediction)

    # Create DataFrame for predictions
    df_pred = pd.DataFrame({'Close Holt-Winters Prediction': predictions}, index=future_dates)
    
    # Calculate in-sample errors
    mae = mean_absolute_error(monthly_df['Close'], model_hw.fittedvalues)
    rmse = np.sqrt(mean_squared_error(monthly_df['Close'], model_hw.fittedvalues))
    mape = np.mean(np.abs((monthly_df['Close'] - model_hw.fittedvalues) / monthly_df['Close'])) * 100
    
    # Create the plot
    fig = go.Figure()

    fig.add_trace(go.Scatter(x=monthly_df.index, y=monthly_df['Close'], mode='lines', name='Close (Historical)', line=dict(color='green')))

    # Add Holt-Winters historical data line (Orange)
    # Assuming you have the fitted values in a column named 'Close Holt-Winters' in your monthly_df
    fig.add_trace(go.Scatter(x=monthly_df.index, y=monthly_df['Close Holt-Winters'], mode='lines', name='Holt-Winters (Historical)', line=dict(color='orange')))  
    
    # Add this trace for future predictions (Blue dashed line)
    fig.add_trace(go.Scatter(x=df_pred.index, y=df_pred['Close Holt-Winters Prediction'], mode='lines', name='Holt-Winters Prediction', line=dict(color='blue', dash='dash'))) 
    

    # Update layout
    fig.update_layout(
        title_text="Close and Holt-Winters Chart (Monthly Aggregation)",
        xaxis_title="Date",
        yaxis_title="Price",
        xaxis_range=[monthly_df.index.min(), df_pred.index.max()],  # Extend x-axis range to include predictions
        height=600,
    )

    # Display parameters and error metrics
    st.write(f"Alpha: {smoothing_level:.2f}, Beta: {smoothing_trend:.2f}, Gamma: {smoothing_seasonal:.2f}")
    st.write(f"**Chỉ số lỗi (Holt Winter):**")
    st.write(f"  - MAE: {mae:.2f}")
    st.write(f"  - RMSE: {rmse:.2f}")
    st.write(f"  - MAPE: {mape:.2f}%")


    return fig, df_pred # R


MODELS_DIR = "models"

def evaluate_model_on_dataset(model_path, scaler_path, X, y_true):
    """Đánh giá chỉ số lỗi của 1 model trên một tập dữ liệu cụ thể (train/test)."""
    model = load_model(model_path)
    scaler = joblib.load(scaler_path)
    
    # Dự đoán
    y_pred = model.predict(X, verbose=0)
    
    # Đảo ngược chuẩn hóa
    y_pred_inv = scaler.inverse_transform(y_pred)
    y_true_inv = scaler.inverse_transform(y_true)

    # Tính toán chỉ số lỗi
    mae = mean_absolute_error(y_true_inv, y_pred_inv)
    rmse = np.sqrt(mean_squared_error(y_true_inv, y_pred_inv))
    mape = np.mean(np.abs((y_true_inv - y_pred_inv) / y_true_inv)) * 100

    return round(mae, 2), round(rmse, 2), round(mape, 2)

def load_all_models_and_compute_errors(X_train, y_train, X_test, y_test, window_size=60):
    """
    Đọc tất cả model trong thư mục models và tính chỉ số lỗi trên cả tập train và test.
    
    Returns:
        df_train_errors (DataFrame): Bảng chỉ số lỗi trên tập Train.
        df_test_errors (DataFrame): Bảng chỉ số lỗi trên tập Test.
    """
    train_results = []
    test_results = []

    if not os.path.exists(MODELS_DIR):
        return pd.DataFrame(), pd.DataFrame()

    model_files = [f for f in os.listdir(MODELS_DIR) if f.endswith("_model.h5")]

    for model_file in model_files:
        model_name = model_file.replace("_model.h5", "")
        model_path = os.path.join(MODELS_DIR, model_file)
        scaler_path = os.path.join(MODELS_DIR, f"{model_name}_scaler.pkl")

        if not os.path.exists(scaler_path):
            continue  # Bỏ qua nếu thiếu scaler tương ứng

        # Kiểm tra đủ dữ liệu cho dự đoán với window_size
        if len(X_train) < window_size or len(X_test) < window_size:
            continue

        # Lấy chuỗi dữ liệu cuối cùng làm input cho model
        X_train_seq = X_train[-1].reshape(1, window_size, 1)  
        X_test_seq = X_test[-1].reshape(1, window_size, 1)


        # Tính lỗi trên tập Train
        mae_train, rmse_train, mape_train = evaluate_model_on_dataset(
            model_path, scaler_path, X_train_seq, y_train[-1].reshape(1, 1)
        )
        train_results.append({
            'Model': model_name,
            'MAE': mae_train,
            'RMSE': rmse_train,
            'MAPE (%)': mape_train
        })

        # Tính lỗi trên tập Test
        mae_test, rmse_test, mape_test = evaluate_model_on_dataset(
            model_path, scaler_path, X_test_seq, y_test[-1].reshape(1, 1)
        )
        test_results.append({
            'Model': model_name,
            'MAE': mae_test,
            'RMSE': rmse_test,
            'MAPE (%)': mape_test
        })

    df_train_errors = pd.DataFrame(train_results)
    df_test_errors = pd.DataFrame(test_results)

    return df_train_errors, df_test_errors

