# Streamlit & Giao diện
import streamlit as st

# Dữ liệu & xử lý dữ liệu
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os
import json

# Trực quan hóa
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.graph_objects as go

# Phân tích thống kê & mô hình chuỗi thời gian
from scipy import stats
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.holtwinters import (
    SimpleExpSmoothing,
    Holt,
    ExponentialSmoothing
)

# Đánh giá mô hình
from sklearn.metrics import mean_absolute_error, mean_squared_error 
import joblib
from streamlit_option_menu import option_menu



# Đánh giá mô hình
from models import (
    create_dataset,
    train_lstm_model_from_csv,
    predict_future_gru,
    predict_future
)

# Import tất cả các hàm từ models_goodnine.py
from models_goodnine import (
    create_dataset_uni,
    create_dataset_multi,
    add_technical_indicators,
    train_dual_branch_model,
    train_model_optimized,
    predict_goodnine,
    predict_full_from_scratch
)


from functions import (
    # Xử lý dữ liệu
    safe_float,
    calculate_statistics,

    # Biểu đồ thống kê
    plot_price_movement_chart,
    plot_candlestick_chart,
    plot_interactive_close_histogram,
    plot_interactive_volume_histogram,
    plot_growth_histogram,
    plot_close_vs_volume_scatter_with_correlation,
    plot_weekday_analysis_chart,
    plot_combined_chart_by_month,
    plot_volume_and_growth_by_weekday,
    plot_volume_and_growth_by_month,
    plot_total_traded_value_by_month,
    plot_total_and_avg_close_combined_by_month,
    plot_total_traded_value_by_quarter,
    plot_close_boxplot,
    plot_interactive_autocorrelation,
    plot_interactive_decomposition,
    plot_rsi,
    plot_log_return,
    plot_price_and_volatility,
    plot_momentum_5,
    plot_macd,
    plot_bollinger_bands,

    # Mô hình dự báo
    create_adj_close_multi_ma_chart_with_prediction,
    calc_ma_prediction_with_real_test,
    create_adj_close_es_chart_with_prediction,
    create_adj_close_holt_chart_with_prediction,
    create_adj_close_holt_winters_chart_with_prediction,
    apply_es_monthly,
    apply_holt_monthly,
    apply_holt_winters_monthly,
)


# PHẢI ĐỂ NGAY ĐÂY!
st.set_page_config(page_title="Stock Prediction", page_icon="📈", layout="wide")

def load_css():
    with open("styles.css", encoding="utf-8") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

load_css()  # Gọi ngay sau khi import thư viện




def main():
    ma_period = None  # Khởi tạo ma_period bằng None
    # Định nghĩa forecast_days ở đây
    forecast_days = 7  # Hoặc bất kỳ giá trị nào bạn muốn


    with st.sidebar:
        # Thêm Google Fonts vào giao diện
        st.markdown("""
        <h1 style='
            text-align:center; 
            font-family:Poppins, Roboto, sans-serif; 
            color:#007bff; 
            font-weight: 700; 
            text-transform: uppercase;
            background: linear-gradient(90deg, #007bff, #00c6ff);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
        '>
        Ứng Dụng Phân Tích <br> Chứng Khoán
        </h1>
        """, unsafe_allow_html=True)

        selected_tab = option_menu(
            menu_title=None,
            options=["Statistics", "Traditional Models", "Machine Learning", "Optimized ML", "Model Performance"],
            icons=["bar-chart", "activity", "cpu", "list"],
            menu_icon="cast",
            default_index=0,
            styles={
                "container": {
                    "padding": "0!important",
                    "background-color": "#E8FCFF",
                    "font-family": "Poppins, Roboto, sans-serif",
                    "border": "none"
                },
                "icon": {
                    "color": "#007bff",
                    "font-size": "20px",
                    "transition": "transform 0.3s ease"
                },
                "nav-link": {
                    "font-size": "18px",
                    "text-align": "left",
                    "margin": "6px auto",  # Căn giữa và có khoảng cách rõ hơn
                    "border-radius": "16px", 
                    "padding": "10px 16px",
                    "width": "85%",  # Giảm chiều rộng để hiển thị rõ bo góc
                    "color": "#333333",
                    "background-color": "#ffffff",
                    "transition": "all 0.3s ease",
                    "font-family": "Poppins, Roboto, sans-serif",
                    "font-weight": "500",
                    "box-shadow": "0 4px 12px rgba(0, 0, 0, 0.05)"
                },
                "nav-link:hover": {
                    "background-color": "#d0f0f8",
                    "color": "#007bff",
                    "transform": "scale(1.05)",
                    "box-shadow": "0 8px 24px rgba(0, 123, 255, 0.2)"
                },
                "nav-link-selected": {
                    "background-color": "#007bff",
                    "color": "#ffffff",
                    "font-weight": "600",
                    "border-radius": "24px",  # Bo tròn nhiều hơn khi chọn
                    "padding": "14px 24px",   # Phóng to tab khi chọn
                    "width": "90%",           # Giữ chiều rộng để viền hiện rõ
                    "box-shadow": "0 12px 32px rgba(0, 123, 255, 0.4)",
                    "transform": "scale(1.1)" # Hiệu ứng phóng to khi chọn
                },
                "icon-color": "#007bff",
                "icon-color-active": "#ffffff"
            }
        )






    # Home Tab
    if selected_tab == "Statistics":
        # Dùng HTML để căn giữa tiêu đề chính
        st.markdown(
        """
        <h1 style="text-align:center;">THỐNG KÊ MÔ TẢ</h1>
        """,
        unsafe_allow_html=True
        )
        st.subheader('THÔNG SỐ ĐẦU VÀO')

        # Lấy danh sách các file CSV trong thư mục dataset
        dataset_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "dataset")
        csv_files = [f for f in os.listdir(dataset_dir) if f.endswith('.csv')]

        import datetime
        col1, col2, col3 = st.columns(3)
        with col1:
            symbol = st.text_input('Mã chứng khoán', 'CRM')
        with col2:
            start_date = st.date_input(
                'Ngày bắt đầu',
                datetime.date(2005, 1, 1),
                min_value=datetime.date(1900, 1, 1),   # Đặt ngày nhỏ nhất
                max_value=datetime.date(2050, 12, 31)  # Đặt ngày lớn nhất
            )
        with col3:
            end_date = st.date_input(
                'Ngày kết thúc',
                datetime.date(2022, 7, 12),
                min_value=datetime.date(1900, 1, 1),
                max_value=datetime.date(2050, 12, 31)
    )


        if st.button('PHÂN TÍCH'):
            # Tạo danh sách mã chứng khoán từ tên file (loại bỏ .csv và viết hoa)
            available_symbols = [f.replace('.csv', '').upper() for f in csv_files]


            if symbol.upper() in available_symbols:
                file_name = f"{symbol.upper()}.csv"
                file_path = os.path.join(dataset_dir, file_name)

                # Đọc dữ liệu từ file CSV tương ứng
                df = pd.read_csv(file_path)
                df['Date'] = pd.to_datetime(df['Date'])
                df.set_index('Date', inplace=True)

                # Chuyển ngày
                start_dt = pd.to_datetime(start_date)
                end_dt = pd.to_datetime(end_date)

                # Lọc dữ liệu
                df_filtered = df[(df.index >= start_dt) & (df.index <= end_dt)].copy()
                df_filtered = df_filtered.reset_index()  # để có cột 'Date'


                # Tính toán chỉ số
                mean_price = df_filtered['Close'].mean()
                max_price = df_filtered['Close'].max()
                max_date = df_filtered.loc[df_filtered['Close'].idxmax(), 'Date'].strftime('%Y-%m-%d')
                min_price = df_filtered['Close'].min()
                min_date = df_filtered.loc[df_filtered['Close'].idxmin(), 'Date'].strftime('%Y-%m-%d')
                avg_growth = df_filtered['Close'].pct_change().mean() * 100
                avg_volatility = (df_filtered['High'] - df_filtered['Low']).mean()
                market_cap_avg = (df_filtered['Close'] * df_filtered['Volume']).mean()
                log_return = np.log(df_filtered['Close'] / df_filtered['Close'].shift(1)).dropna()
                avg_log_return = log_return.mean()
                std_dev = df_filtered['Close'].std()
                coef_variation = std_dev / mean_price

                # CSS cho metric cards
                st.markdown("""
                <style>
                .metric-card {
                    border: 1px solid #e0e0e0;
                    border-radius: 12px;
                    padding: 20px;
                    text-align: center;
                    background-color: #ffffff;
                    box-shadow: 0 4px 8px rgba(0, 0, 0, 0.05);
                    margin-bottom: 15px;
                }
                .metric-label {
                    font-weight: 600;
                    font-size: 18px;
                    color: #444;
                    margin-bottom: 8px;
                }
                .metric-value {
                    font-size: 24px;
                    font-weight: bold;
                    color: #1a73e8;
                }
                </style>
                """, unsafe_allow_html=True)


            st.subheader(f"KẾT QUẢ PHÂN TÍCH: `{symbol.upper()}`")

            with st.expander("**THỐNG KÊ MÔ TẢ**", expanded=True):
                st.markdown("<div style='font-size:16px; font-weight:600;'>Chỉ số thống kê đáng chú ý</div>", unsafe_allow_html=True)

                # Hàng 1
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.markdown(f"""<div class="metric-card">
                        <div class="metric-label">Giá trung bình</div>
                        <div class="metric-value">{mean_price:.2f}</div>
                    </div>""", unsafe_allow_html=True)
                with col2:
                    st.markdown(f"""<div class="metric-card">
                        <div class="metric-label">Giá cao nhất</div>
                        <div class="metric-value">{max_price:.2f} ({max_date})</div>
                    </div>""", unsafe_allow_html=True)
                with col3:
                    st.markdown(f"""<div class="metric-card">
                        <div class="metric-label">Giá thấp nhất</div>
                        <div class="metric-value">{min_price:.2f} ({min_date})</div>
                    </div>""", unsafe_allow_html=True)
                with col4:
                    st.markdown(f"""<div class="metric-card">
                        <div class="metric-label">Vốn hóa TB</div>
                        <div class="metric-value">{market_cap_avg:,.0f}</div>
                    </div>""", unsafe_allow_html=True)

                # Hàng 2
                col5, col6, col7, col8 = st.columns(4)
                with col5:
                    st.markdown(f"""<div class="metric-card">
                        <div class="metric-label">Trung bình tăng trưởng</div>
                        <div class="metric-value">{avg_growth:.2f}%</div>
                    </div>""", unsafe_allow_html=True)

                with col6:
                    st.markdown(f"""<div class="metric-card">
                        <div class="metric-label">Trung bình biến động giá</div>
                        <div class="metric-value">{avg_volatility:.2f}</div>
                    </div>""", unsafe_allow_html=True)

                with col7:
                    st.markdown(f"""<div class="metric-card">
                        <div class="metric-label">Log return trung bình</div>
                        <div class="metric-value">{avg_log_return:.4f}</div>
                    </div>""", unsafe_allow_html=True)


                with col8:
                    st.markdown(f"""<div class="metric-card">
                        <div class="metric-label">Hệ số biến thiên</div>
                        <div class="metric-value">{coef_variation:.2f}</div>
                    </div>""", unsafe_allow_html=True)

                st.markdown("<div style='font-size:16px; font-weight:600;'>Tham số cơ bản</div>", unsafe_allow_html=True)
                stats_df = calculate_statistics(df, start_date, end_date)
                st.dataframe(stats_df, use_container_width=True)

                # Hai biểu đồ Histogram (Close và Volume) ở hàng trên, chia 2 cột
                col1, col2 = st.columns(2)

                with col1:
                    st.markdown("<div style='font-size:16px; font-weight:600;'>Biểu đồ Histogram giá đóng cửa</div>", unsafe_allow_html=True)
                    plot_interactive_close_histogram(df_filtered)

                with col2:
                    st.markdown("<div style='font-size:16px; font-weight:600;'>Biểu đồ Histogram số lượng giao dịch</div>", unsafe_allow_html=True)
                    plot_interactive_volume_histogram(df_filtered)

                # Biểu đồ Histogram Tỷ lệ tăng trưởng ở hàng dưới, full width
                st.markdown("<div style='font-size:16px; font-weight:600; margin-top:20px;'>Biểu đồ Histogram Tỷ lệ tăng trưởng</div>", unsafe_allow_html=True)
                fig_growth_hist = plot_growth_histogram(df_filtered)
                st.plotly_chart(fig_growth_hist, use_container_width=True)



                st.markdown("<div style='font-size:16px; font-weight:600;'>Biểu đồ Phân Tán & Tương Quan: Close vs Volume</div>", unsafe_allow_html=True)
                plot_close_vs_volume_scatter_with_correlation(df_filtered)

                st.markdown("<div style='font-size:16px; font-weight:600;'>Biểu đồ hộp phân phối giá đóng cửa (Close)</div>", unsafe_allow_html=True)
                plot_close_boxplot(df_filtered)


                st.markdown("<div style='font-size:16px; font-weight:600;'>Chỉ báo RSI – Động lượng giá cổ phiếu</div>", unsafe_allow_html=True)
                plot_rsi(df_filtered)

                st.markdown("<div style='font-size:16px; font-weight:600;'>Chỉ báo MACD</div>", unsafe_allow_html=True)
                plot_macd(df_filtered)

                st.markdown("<div style='font-size:16px; font-weight:600;'>Chỉ báo Log Return</div>", unsafe_allow_html=True)
                plot_log_return(df_filtered)


                st.markdown("<div style='font-size:16px; font-weight:600;'>Chỉ báo Volatility</div>", unsafe_allow_html=True)
                plot_price_and_volatility(df_filtered)


                st.markdown("<div style='font-size:16px; font-weight:600;'>Chỉ báo Momentum</div>", unsafe_allow_html=True)
                plot_momentum_5(df_filtered)


                st.markdown("<div style='font-size:16px; font-weight:600;'>Chỉ báo Bollinger Bands</div>", unsafe_allow_html=True)
                plot_bollinger_bands(df_filtered)




            with st.expander("📌 **PHÂN TÍCH CHUỖI THỜI GIAN**", expanded=True):

                st.markdown("<div style='font-size:16px; font-weight:600;'>Biều đồ biến động giá đóng cửa và số lượng giao dịch</div>", unsafe_allow_html=True)
                fig_price = plot_price_movement_chart(df, start_date, end_date)
                st.plotly_chart(fig_price, use_container_width=True)

                st.markdown("<div style='font-size:16px; font-weight:600;'>Biểu đồ Nến Giá Cổ Phiếu</div>", unsafe_allow_html=True)
                fig_candle = plot_candlestick_chart(df, start_date, end_date)
                st.plotly_chart(fig_candle, use_container_width=True)

                col1, col2 = st.columns(2)

                with col1:
                    st.markdown("<div style='font-size:16px; font-weight:600;'>Biểu đồ Tổng giá trị giao dịch theo tháng (Close × Volume)</div>", unsafe_allow_html=True)
                    plot_total_traded_value_by_month(df_filtered)

                with col2:
                    st.markdown("<div style='font-size:16px; font-weight:600;'>Biểu đồ Tổng giá trị giao dịch theo quý (Close × Volume)</div>", unsafe_allow_html=True)
                    plot_total_traded_value_by_quarter(df_filtered)


                st.markdown("<div style='font-size:16px; font-weight:600;'>Biểu đồ Tổng & Trung bình giá đóng cửa theo tháng</div>", unsafe_allow_html=True)
                plot_total_and_avg_close_combined_by_month(df_filtered)


                st.markdown("<div style='font-size:16px; font-weight:600;'>Biểu đồ phân tích theo thứ trong tuần</div>", unsafe_allow_html=True)
                col1, col2 = st.columns(2)

                with col1:
                    st.markdown("**📈 Giá đóng cửa & tăng trưởng**")
                    plot_weekday_analysis_chart(df_filtered)

                with col2:
                    st.markdown("**📊 Khối lượng giao dịch & tăng trưởng**")
                    plot_volume_and_growth_by_weekday(df_filtered)


                # ======= BIỂU ĐỒ THEO THÁNG TRONG NĂM =======
                st.markdown("<div style='font-size:16px; font-weight:600;'>Biểu đồ phân tích theo tháng trong năm</div>", unsafe_allow_html=True)
                col3, col4 = st.columns(2)

                with col3:
                    st.markdown("**📈 Giá đóng cửa & tăng trưởng**")
                    plot_combined_chart_by_month(df_filtered)

                with col4:
                    st.markdown("**📊 Khối lượng giao dịch & tăng trưởng**")
                    plot_volume_and_growth_by_month(df_filtered)


                st.markdown("<div style='font-size:16px; font-weight:600;'>Biểu đồ Tự tương quan</div>", unsafe_allow_html=True)
                # Hiển thị biểu đồ
                plot_interactive_autocorrelation(df_filtered)

                st.markdown("<div style='font-size:16px; font-weight:600;'>Biểu đồ phân tích thành phần chuỗi thời gian (Time Series Decomposition)</div>", unsafe_allow_html=True)
                # Hiển thị biểu đồ
                plot_interactive_decomposition(df_filtered)

                







    # Prediction Tab
    elif selected_tab == "Traditional Models":
        # Dùng HTML để căn giữa tiêu đề chính
        st.markdown(
        """
        <h1 style="text-align:center;">PHÂN TÍCH MÔ HÌNH DỰ ĐOÁN TRUYỀN THỐNG</h1>
        """,
        unsafe_allow_html=True
        )



        # Lấy danh sách các file CSV trong thư mục dataset
        dataset_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "dataset")
        csv_files = [f for f in os.listdir(dataset_dir) if f.endswith('.csv')]

        selected_symbol = st.text_input("Nhập mã chứng khoán", "MSFT", key="selected_symbol").upper().strip()
        selected_file = f"{selected_symbol}.csv"

        # Kiểm tra xem người dùng đã chọn file hay chưa
        if selected_file:
            # Đường dẫn tới file CSV đã chọn
            file_path = os.path.join(dataset_dir, selected_file)

            # Đọc dữ liệu từ file CSV đã chọn
            df = pd.read_csv(file_path)

        # Chọn mô hình dự báo
        st.subheader("MÔ HÌNH DỰ ĐOÁN")
        model_choice = st.selectbox("Mô hình:",
                                    ["Simple Moving Average",
                                    "Exponential Smoothing By Day",
                                    "Exponential Smoothing By Month",
                                    "Holt By Day", "Holt By Month",
                                    "Holt Winter By Day", "Holt Winter By Month"
                                    ])

        # Chọn thời gian dự đoán (chỉ cho Simple Moving Average)
        if model_choice == "Simple Moving Average":
            st.subheader("THỜI GIAN DỰ ĐOÁN")

            # 1. Nhập nhiều kỳ hạn MA, ví dụ: 20,50,100
            ma_input = st.text_input("Nhập các kỳ hạn MA", "20,50,100", key="ma_input")


            # Chuyển thành list int, lọc số hợp lệ
            ma_windows = [int(x.strip()) for x in ma_input.split(",") if x.strip().isdigit() and int(x.strip()) > 0]
            if not ma_windows:
                st.warning("Vui lòng nhập ít nhất 1 kỳ hạn MA hợp lệ.")

            # 2. Chọn thời gian dự đoán
            forecast_period = st.selectbox(
                "Thời gian dự đoán:",
                ["1 ngày", "1 tuần (5 ngày)", "1 tháng (22 ngày)", "Khác"]
            )

            if forecast_period == "Khác":
                custom_days = st.number_input("Nhập số ngày dự đoán:", min_value=1, value=1)
                forecast_days = custom_days
            else:
                forecast_days = {
                    "1 ngày": 1,
                    "1 tuần (5 ngày)": 5,
                    "1 tháng (22 ngày)": 22,
                }[forecast_period]


        elif model_choice == "Exponential Smoothing By Day":
            st.subheader("Chọn thời gian dự đoán:")
            forecast_period = st.selectbox("Thời gian:",
                                                ["1 ngày", "1 tuần (5 ngày)",
                                                "1 tháng (22 ngày)", "Khác"])
            
             # Trong phần Pred, thêm thanh điều chỉnh:
            smoothing_level = st.slider("Alpha (Smoothing Level)", 0.01, 1.0, 0.1, 0.01)

            # Nếu chọn "Khác", cho phép nhập số ngày dự đoán
            if forecast_period == "Khác":
                ma_period = st.number_input("Nhập số ngày dự đoán:", min_value=1, value=1, step=1)
                # Đảm bảo không bị None (number_input luôn trả về số, không None, trừ khi code khác tác động)
            else:
                ma_period = {
                    "1 ngày": 1,
                    "1 tuần (5 ngày)": 5,
                    "1 tháng (22 ngày)": 22,
                }[forecast_period]

            try:
                ma_period = int(ma_period)
            except Exception:
                st.error("Vui lòng nhập số ngày dự đoán hợp lệ!")
                st.stop()


        elif model_choice == "Exponential Smoothing By Month":
            st.subheader("Chọn thời gian dự đoán:")

            seasonality_periods = st.number_input("Giai đoạn mùa vụ", min_value=1, value=12, step=1)

            forecast_period = st.selectbox("Thời gian:",
                                                ["1 tháng", "6 tháng",
                                                "12 tháng", "Khác"])
            
             # Trong phần Pred, thêm thanh điều chỉnh:
            alpha_es = st.slider("Alpha (Smoothing Level)", 0.01, 1.0, 0.1, 0.01)

            # Nếu chọn "Khác", cho phép nhập số ngày dự đoán
            if forecast_period == "Khác":
                custom_days = st.number_input("Nhập số ngày dự đoán:", min_value=1, value=1)
                ma_period = custom_days  # Gán custom_days cho ma_period nếu chọn "Khác"
            else:
                # Gán ma_period dựa trên forecast_period đã chọn
                ma_period = {
                    "1 tháng": 1,
                    "6 tháng": 6,
                    "12 tháng": 12,
                }[forecast_period]


        elif model_choice == "Holt By Day":
            st.subheader("Chọn thời gian dự đoán:")
            forecast_period = st.selectbox("Thời gian:",
                                                ["1 ngày", "1 tuần (5 ngày)",
                                                "1 tháng (22 ngày)", "Khác"])

            st.subheader("Chọn hệ số alpha và beta:")
            alpha = st.slider("Alpha (Smoothing Level):", min_value=0.01, max_value=1.0, value=0.1, step=0.01)
            beta = st.slider("Beta (Smoothing Trend):", min_value=0.01, max_value=1.0, value=0.2, step=0.01)


            # Nếu chọn "Khác", cho phép nhập số ngày dự đoán
            if forecast_period == "Khác":
                custom_days = st.number_input("Nhập số ngày dự đoán:", min_value=1, value=1)
                ma_period = custom_days  # Gán custom_days cho ma_period nếu chọn "Khác"
            else:
                # Gán ma_period dựa trên forecast_period đã chọn
                ma_period = {
                    "1 ngày": 1,
                    "1 tuần (5 ngày)": 5,
                    "1 tháng (22 ngày)": 22,
                }[forecast_period]

        elif model_choice == "Holt By Month":
            st.subheader("Chọn thời gian dự đoán:")

            seasonality_periods = st.number_input("Giai đoạn mùa vụ", min_value=1, value=12, step=1)

            forecast_period = st.selectbox("Thời gian:",
                                                ["1 tháng", "6 tháng",
                                                "12 tháng", "Khác"])

            # Add sliders for Holt-Winters parameters
            st.subheader("Holt-Winters Parameters")
            alpha_holt = st.slider("Smoothing Level (Alpha)", 0.01, 1.0, 0.2, 0.01)
            beta_holt = st.slider("Smoothing Trend (Beta)", 0.01, 1.0, 0.1, 0.01)


            # Nếu chọn "Khác", cho phép nhập số ngày dự đoán
            if forecast_period == "Khác":
                custom_days = st.number_input("Nhập số ngày dự đoán:", min_value=1, value=1)
                ma_period = custom_days  # Gán custom_days cho ma_period nếu chọn "Khác"
            else:
                # Gán ma_period dựa trên forecast_period đã chọn
                ma_period = {
                    "1 tháng": 1,
                    "6 tháng": 6,
                    "12 tháng": 12,
                }[forecast_period]

        elif model_choice == "Holt Winter By Day":
            st.subheader("Chọn thời gian dự đoán:")

            seasonality_periods = st.number_input("Giai đoạn mùa vụ", min_value=1, value=252, step=1)

            forecast_period = st.selectbox("Thời gian:",
                                                ["1 ngày", "1 tuần (5 ngày)",
                                                "1 tháng (22 ngày)", "Khác"])

            st.subheader("Chọn hệ số alpha và beta:")
            alpha = st.slider("Alpha (Smoothing Level):", min_value=0.01, max_value=1.0, value=0.1, step=0.01)
            beta = st.slider("Beta (Smoothing Trend):", min_value=0.01, max_value=1.0, value=0.2, step=0.01)
            gamma = st.slider("Gamma (Smoothing Seasonality):", min_value=0.01, max_value=1.0, value=0.2, step=0.01)


            # Nếu chọn "Khác", cho phép nhập số ngày dự đoán
            if forecast_period == "Khác":
                custom_days = st.number_input("Nhập số ngày dự đoán:", min_value=1, value=1)
                ma_period = custom_days  # Gán custom_days cho ma_period nếu chọn "Khác"
            else:
                # Gán ma_period dựa trên forecast_period đã chọn
                ma_period = {
                    "1 ngày": 1,
                    "1 tuần (5 ngày)": 5,
                    "1 tháng (22 ngày)": 22,
                }[forecast_period]

        elif model_choice == "Holt Winter By Month":
            st.subheader("Chọn thời gian dự đoán:")

            seasonality_periods = st.number_input("Giai đoạn mùa vụ", min_value=1, value=12, step=1)

            forecast_period = st.selectbox("Thời gian:",
                                                ["1 tháng", "6 tháng",
                                                "12 tháng", "Khác"])

            # Add sliders for Holt-Winters parameters
            st.subheader("Holt-Winters Parameters")
            alpha_hwm = st.slider("Smoothing Level (Alpha)", 0.01, 1.0, 0.2, 0.01)
            beta_hwm = st.slider("Smoothing Trend (Beta)", 0.01, 1.0, 0.1, 0.01)
            gamma_hwm = st.slider("Smoothing Seasonal (Gamma)", 0.01, 1.0, 0.1, 0.01)


            # Nếu chọn "Khác", cho phép nhập số ngày dự đoán
            if forecast_period == "Khác":
                custom_days = st.number_input("Nhập số ngày dự đoán:", min_value=1, value=1)
                ma_period = custom_days  # Gán custom_days cho ma_period nếu chọn "Khác"
            else:
                # Gán ma_period dựa trên forecast_period đã chọn
                ma_period = {
                    "1 tháng": 1,
                    "6 tháng": 6,
                    "12 tháng": 12,
                }[forecast_period]

        # Nút Dự báo
        selected_file = f"{selected_symbol}.csv"

        if st.button('DỰ ĐOÁN'):
            # ==== BẮT ĐẦU SPINNER ====
            with st.spinner("Đang dự đoán, vui lòng chờ..."):
                file_path = os.path.join(dataset_dir, selected_file)
                df = pd.read_csv(file_path)
                df['Date'] = pd.to_datetime(df['Date'])
                df.set_index('Date', inplace=True)

                cutoff_date = pd.to_datetime('2022-07-12')
                df_train = df[df.index <= cutoff_date]

                if model_choice == "Simple Moving Average":
                    ma_windows = [int(x.strip()) for x in ma_input.split(",") if x.strip().isdigit() and int(x.strip()) > 0]
                    if not ma_windows:
                        st.warning("Vui lòng nhập ít nhất 1 kỳ hạn MA hợp lệ.")
                    else:
                        prediction_tables, error_df = calc_ma_prediction_with_real_test(
                            df,
                            ma_windows=ma_windows,
                            forecast_days=forecast_days,
                            train_ratio=0.8
                        )

                        # Đọc dữ liệu future thực tế từ yfinance nếu có
                        yf_path = os.path.join("D:/Data Science/yfinance", f"{selected_symbol}.csv")
                        future_actual_dates = []
                        future_actual_close = []
                        if os.path.exists(yf_path):
                            df_yf = pd.read_csv(yf_path)
                            df_yf['Date'] = pd.to_datetime(df_yf['Date'])
                            df_yf = df_yf.sort_values('Date').reset_index(drop=True)
                            future_actual_dates = df_yf['Date'].iloc[:forecast_days]
                            future_actual_close = df_yf['Close'].iloc[:forecast_days]

                        # Vẽ biểu đồ cho từng MA
                        for window, preds in prediction_tables.items():
                            fig = go.Figure()

                            fig.add_trace(go.Scatter(
                                x=preds["Train Dates"], y=preds["Train Actual"],
                                name="Train Actual", line=dict(color='#1E90FF')
                            ))
                            fig.add_trace(go.Scatter(
                                x=preds["Test Dates"], y=preds["Test Actual"],
                                name="Test Actual", line=dict(color='black')
                            ))
                            fig.add_trace(go.Scatter(
                                x=preds["Test Dates"], y=preds["Test Predict"],
                                name=f"MA{window} - Test Predict", line=dict(color='red', dash='dash')
                            ))
                            fig.add_trace(go.Scatter(
                                x=preds["Future Dates"], y=preds["Future Predict"],
                                name=f"MA{window} - Future Forecast", line=dict(color='purple', dash='dot')
                            ))

                            # Thêm Future Actual nếu có
                            if len(future_actual_dates) > 0:
                                fig.add_trace(go.Scatter(
                                    x=future_actual_dates, y=future_actual_close,
                                    name="Future Actual", line=dict(color='green', width=2)
                                ))

                            fig.update_layout(
                                title=f"Biểu đồ MA{window} - Train/Test/Forecast ({selected_symbol})",
                                xaxis_title="Ngày", yaxis_title="Giá trị",
                                plot_bgcolor='white',
                                xaxis=dict(showgrid=False),
                                yaxis=dict(showgrid=False),
                                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                                margin=dict(l=40, r=40, t=60, b=40)
                            )

                            st.plotly_chart(fig, use_container_width=True, key=f"ma_plot_{window}")

                        st.subheader("Bảng chỉ số lỗi Train/Test/Future của từng MA:")
                        st.dataframe(error_df)




                elif model_choice == "Exponential Smoothing By Day":
                    ma_period = int(ma_period)
                    fig, test_pred_df, future_pred_df, error_df = create_adj_close_es_chart_with_prediction(
                        df,
                        smoothing_level=smoothing_level,   # lấy từ st.slider()
                        train_ratio=0.8,                   # hoặc bạn tự chọn tỉ lệ
                        test_ratio=0.2,
                        forecast_days=ma_period,
                        symbol=selected_symbol,
                        test_folder="D:/Data Science/yfinance/"
                    )
                    st.plotly_chart(fig, use_container_width=True)
                    st.caption(f"Tham số Alpha: {smoothing_level:.2f}")
                    st.subheader("Bảng dự đoán Test (so sánh giá thật và dự báo):")
                    st.dataframe(test_pred_df)
                    if future_pred_df is not None:
                        st.subheader("Bảng dự đoán Future:")
                        st.dataframe(future_pred_df)
                    st.subheader("Bảng chỉ số lỗi (Train/Test/Future):")
                    st.dataframe(error_df)



                elif model_choice == "Exponential Smoothing By Month":

                    # Call the ES monthly function
                    fig_es_monthly, df_pred_es_monthly, mae, rmse, mape = apply_es_monthly(df, alpha_es, ma_period)

                    # Hiển thị chỉ số lỗi
                    st.write(f"Alpha: {alpha_es:.2f}%")
                    st.write(f"**Chỉ số lỗi (ES):**")
                    st.write(f"  - MAE: {mae:.2f}")
                    st.write(f"  - RMSE: {rmse:.2f}")
                    st.write(f"  - MAPE: {mape:.2f}%")

                    # Display the chart and prediction table
                    st.plotly_chart(fig_es_monthly, use_container_width=True)
                    st.subheader("Bảng dự đoán ES (Monthly):")
                    st.dataframe(df_pred_es_monthly)  # Display the prediction DataFrame

                elif model_choice == "Holt By Day":
                    forecast_days = int(ma_period)  # Số ngày dự báo tương lai

                    fig, test_pred_df, future_pred_df, error_df = create_adj_close_holt_chart_with_prediction(
                        df,
                        smoothing_level=alpha,
                        smoothing_slope=beta,
                        train_ratio=0.8,
                        test_ratio=0.2,
                        forecast_days=forecast_days,
                        symbol=selected_symbol,
                        test_folder="D:/Data Science/yfinance/"
                    )

                    st.plotly_chart(fig, use_container_width=True)

                    st.caption(f"Tham số: Alpha={alpha:.2f}, Beta={beta:.2f}")

                    st.subheader("Bảng dự đoán Test (so sánh giá thật và dự báo):")
                    st.dataframe(test_pred_df)

                    if future_pred_df is not None:
                        st.subheader("Bảng dự đoán Future:")
                        st.dataframe(future_pred_df)

                    st.subheader("Bảng chỉ số lỗi (Train/Test/Future):")
                    st.dataframe(error_df)



                elif model_choice == "Holt By Month":
                # Call the Holt-Winters monthly function
                    fig_holt_monthly, df_pred_holt_monthly = apply_holt_monthly(
                        df,
                        smoothing_level=alpha_holt,
                        smoothing_trend=beta_holt,
                        forecast_days=ma_period
                    )

                    # Display the chart and prediction table
                    st.plotly_chart(fig_holt_monthly, use_container_width=True)
                    st.subheader("Bảng dự đoán Holt (Monthly):")
                    st.dataframe(df_pred_holt_monthly)  

                elif model_choice == "Holt Winter By Day":
                    forecast_days = int(ma_period)

                    fig, test_pred_df, future_pred_df, error_df = create_adj_close_holt_winters_chart_with_prediction(
                        df,
                        alpha=alpha,
                        beta=beta,
                        gamma=gamma,
                        seasonality_periods=seasonality_periods,
                        train_ratio=0.8,
                        test_ratio=0.2,
                        forecast_days=forecast_days,
                        symbol=selected_symbol,
                        test_folder="D:/Data Science/yfinance/"
                    )

                    st.plotly_chart(fig, use_container_width=True)
                    st.caption(f"Tham số: Alpha={alpha:.2f}, Beta={beta:.2f}, Gamma={gamma:.2f}, Chu kỳ mùa vụ={seasonality_periods}")

                    st.subheader("Bảng dự đoán Test (so sánh giá thật và dự báo):")
                    st.dataframe(test_pred_df)

                    if future_pred_df is not None:
                        st.subheader("Bảng dự đoán Future:")
                        st.dataframe(future_pred_df)

                    st.subheader("Bảng chỉ số lỗi (Train/Test/Future):")
                    st.dataframe(error_df)


                elif model_choice == "Holt Winter By Month":
                # Call the Holt-Winters monthly function
                    fig_hwm, df_pred_hwm = apply_holt_winters_monthly(
                        df,
                        smoothing_level=alpha_hwm,
                        smoothing_trend=beta_hwm,
                        smoothing_seasonal=gamma_hwm,
                        forecast_days=ma_period
                    )
                    
                    # Display the chart and prediction table
                    st.plotly_chart(fig_hwm, use_container_width=True)
                    st.subheader("Bảng dự đoán Holt-Winters (Monthly):")
                    st.dataframe(df_pred_hwm)

           

    elif selected_tab == "Machine Learning":
        from tensorflow.keras.models import load_model
        st.markdown("<h1 style='text-align:center;'>MACHINE LEARNING</h1>", unsafe_allow_html=True)
        st.subheader("HUẤN LUYỆN MÔ HÌNH")
        st.subheader("Chọn mô hình huấn luyện")
        model_type = st.selectbox("Loại mô hình:", ["LSTM", "GRU"])
        # Chọn file dữ liệu
        dataset_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "dataset")
        selected_symbol = st.text_input("Nhập mã chứng khoán", "AAPL").upper().strip()
        selected_file = f"{selected_symbol}.csv"
        file_path = os.path.join(dataset_dir, selected_file)

        # Tham số
        window_size = st.slider("Sliding window", 10, 1000, 200, 5)
        epochs = st.slider("Epochs", 5, 100, 20)
        batch_size = st.slider("Batch size", 8, 512, 64)

        # ===== NÚT HUẤN LUYỆN =====
        if st.button("HUẤN LUYỆN"):
            with st.spinner("Đang huấn luyện..."):

                # === Đường dẫn ===
                model_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")
                errors_dir = os.path.join(os.path.dirname(model_dir), "errors")
                os.makedirs(model_dir, exist_ok=True)
                os.makedirs(errors_dir, exist_ok=True)

                # === Xác định tên file mô hình .h5 cần tìm ===
                model_file = f"{selected_symbol}_model.h5" if model_type == "LSTM" else f"{selected_symbol}_gru_model.h5"
                model_path = os.path.join(model_dir, model_file)

                # === Kiểm tra mô hình đã tồn tại hay chưa ===
                is_first_time = not os.path.exists(model_path)
                old_mape = None

                # Nếu đã có mô hình, kiểm tra file lỗi cũ
                if not is_first_time:
                    error_suffix = "errors.csv" if model_type == "LSTM" else "gru_errors.csv"
                    error_file = os.path.join(errors_dir, f"{selected_symbol}_{error_suffix}")

                    if os.path.exists(error_file):
                        try:
                            df_errors = pd.read_csv(error_file)
                            if not df_errors.empty:
                                old_mape = df_errors.iloc[-1]['MAPE_Test (%)']
                        except:
                            old_mape = None

                # === Tiến hành huấn luyện ===
                if model_type == "LSTM":
                    results = train_lstm_model_from_csv(file_path, window_size, epochs, batch_size)
                else:
                    results = train_gru_model_from_csv(file_path, window_size, epochs, batch_size)

                new_mape = results['mape_test']

                # === Thông báo sau huấn luyện ===
                st.success(f"ĐÃ HUẤN LUYỆN THÀNH CÔNG MÔ HÌNH {model_type} CHO MÃ `{selected_symbol}`!")
                st.info(f"""
                ### Thông tin mô hình:
                - Loại mô hình: `{model_type}`
                - Mã chứng khoán: `{selected_symbol}`
                - Sliding Window: `{window_size}`  
                - Epochs: `{epochs}`  
                - Batch Size: `{batch_size}`  
                """)

                # === So sánh và thông báo cải tiến ===
                if is_first_time or old_mape is None:
                    st.success(f"Đây là mô hình đầu tiên cho mã `{selected_symbol}`. Đã lưu thành công!")
                else:
                    if new_mape < old_mape:
                        st.success(f"Mô hình mới tốt hơn mô hình cũ (MAPE Test giảm từ `{old_mape:.2f}%` xuống `{new_mape:.2f}%`). Đã cập nhật!")
                    elif new_mape > old_mape:
                        st.warning(f"Mô hình mới có MAPE Test `{new_mape:.2f}%` cao hơn mô hình cũ `{old_mape:.2f}%`. Đang giữ mô hình cũ.")
                    else:
                        st.info(f"Mô hình mới có MAPE Test bằng mô hình cũ ({new_mape:.2f}%).")



                st.subheader("Biểu đồ Train & Test")

                fig = go.Figure()

                # Tập Train
                fig.add_trace(go.Scatter(x=results['train_dates'], y=results['y_train_real'].flatten(), name="Giá thực tế (Train)", line=dict(color="blue")))
                fig.add_trace(go.Scatter(x=results['train_dates'], y=results['y_train_pred'].flatten(), name="Dự đoán (Train)", line=dict(color="orange", dash="dot")))

                # Tập Test
                fig.add_trace(go.Scatter(x=results['test_dates'], y=results['y_test_real'].flatten(), name="Giá thực tế (Test)", line=dict(color="green")))
                fig.add_trace(go.Scatter(x=results['test_dates'], y=results['y_test_pred'].flatten(), name="Dự đoán (Test)", line=dict(color="red", dash="dash")))

                # Tùy chỉnh layout
                fig.update_layout(
                    title=f"Dự đoán giá cổ phiếu {selected_symbol} ({model_type})",
                    xaxis_title="Ngày",
                    yaxis_title="Giá trị",
                    plot_bgcolor="white",
                    xaxis=dict(showgrid=False),
                    yaxis=dict(showgrid=False),
                    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                    margin=dict(l=40, r=40, t=80, b=40)
                )

                st.plotly_chart(fig, use_container_width=True)


                st.markdown("### ĐÁNH GIÁ MÔ HÌNH")

                col1, col2 = st.columns(2)

                with col1:
                    st.markdown("Tập Train")
                    st.metric("MAE", f"{results['mae_train']:.2f}")
                    st.metric("RMSE", f"{results['rmse_train']:.2f}")
                    st.metric("MAPE(%)", f"{results['mape_train']:.2f}") 

                with col2:
                    st.markdown("Tập Test")
                    st.metric("MAE", f"{results['mae_test']:.2f}")  
                    st.metric("RMSE", f"{results['rmse_test']:.2f}")
                    st.metric("MAPE(%)", f"{results['mape_test']:.2f}")

                st.markdown("---")
                st.markdown("### SO SÁNH GIÁ TRAIN & TEST VỚI THỰC TẾ (15 dòng gần nhất)")

                col1, col2 = st.columns(2)

                with col1:
                    st.markdown("Train")
                    df_train = pd.DataFrame({
                        "Ngày": results['train_dates'].reset_index(drop=True),
                        "Giá thực tế": results['y_train_real'].flatten(),
                        "Dự đoán": results['y_train_pred'].flatten()
                    })
                    st.dataframe(df_train.tail(15), use_container_width=True)

                with col2:
                    st.markdown("Test")
                    df_test = pd.DataFrame({
                        "Ngày": results['test_dates'].reset_index(drop=True),
                        "Giá thực tế": results['y_test_real'].flatten(),
                        "Dự đoán": results['y_test_pred'].flatten()
                    }) 

                    st.dataframe(df_test.tail(15), use_container_width=True)

            # ===== NÚT DỰ ĐOÁN =====
        st.title("DỰ ĐOÁN TỪ MÔ HÌNH")

        # 1. Lấy thư mục gốc tuyệt đối
        base_dir = os.path.dirname(os.path.abspath(__file__))
        dataset_dir = os.path.join(base_dir, "dataset")
        model_dir = os.path.join(base_dir, "models")
        os.makedirs(dataset_dir, exist_ok=True)
        os.makedirs(model_dir, exist_ok=True)

                # Lấy danh sách mô hình
        if model_type == "LSTM":
            model_files = [f for f in os.listdir(model_dir) if f.endswith("_model.h5")]
            trained_symbols = [f.replace("_model.h5", "") for f in model_files]
        else:
            model_files = [f for f in os.listdir(model_dir) if f.endswith("_gru_model.h5")]
            trained_symbols = [f.replace("_gru_model.h5", "") for f in model_files]

        # Tạo danh sách biểu tượng chuẩn xác
        if model_type == "LSTM":
            model_files = [f for f in os.listdir(model_dir) if f.endswith("_model.h5") and not f.endswith("_gru_model.h5")]
            trained_symbols = [f.replace("_model.h5", "") for f in model_files]
        else:
            model_files = [f for f in os.listdir(model_dir) if f.endswith("_gru_model.h5")]
            trained_symbols = [f.replace("_gru_model.h5", "") for f in model_files]

        # Nếu không có mô hình nào
        if not trained_symbols:
            st.warning(f"Không có mô hình {model_type} nào đã huấn luyện.")
            st.stop()

        selected_symbol = st.selectbox("Chọn mã chứng khoán đã huấn luyện", trained_symbols)

        dataset_path = os.path.join(dataset_dir, f"{selected_symbol}.csv")

        # Đường dẫn model & metadata
        if model_type == "LSTM":
            model_path = os.path.join(model_dir, f"{selected_symbol}_model.h5")
            scaler_path = os.path.join(model_dir, f"{selected_symbol}_scaler.pkl")
        else:
            model_path = os.path.join(model_dir, f"{selected_symbol}_gru_model.h5")
            scaler_path = os.path.join(model_dir, f"{selected_symbol}_gru_scaler.pkl")

        metadata_path = os.path.join(
            model_dir,
            f"{selected_symbol}_metadata.json" if model_type == "LSTM" else f"{selected_symbol}_gru_metadata.json"
        )

            # 4. Nhập tham số dự đoán
        steps = st.number_input("Số ngày muốn dự đoán", min_value=1, max_value=600, value=22, step=1)


        if st.button("DỰ ĐOÁN"):
            with st.spinner('Đang dự đoán, vui lòng chờ...'):
                try:
                    # Load model và scaler
                    model = load_model(model_path)
                    # ✅ Load đúng window_size đã huấn luyện từ metadata
                    if os.path.exists(metadata_path):
                        with open(metadata_path, "r") as f:
                            metadata = json.load(f)
                        window_size = metadata.get("window_size", model.input_shape[1])  # <-- lỗi nếu model chưa load!
                    else:
                        window_size = model.input_shape[1]  # <-- lỗi nếu model chưa load!
                    model = load_model(model_path)  # <-- load model sau!


                    scaler = joblib.load(scaler_path)
                    


                    if not os.path.exists(dataset_path):
                        st.error(f"Không tìm thấy file dữ liệu: `{dataset_path}`.")
                        st.stop()

                    df = pd.read_csv(dataset_path)
                    df['Date'] = pd.to_datetime(df['Date'])
                    df = df.sort_values('Date')

                    if len(df) < window_size:
                        st.error(f"Dữ liệu không đủ để dự đoán. Cần ít nhất {window_size} bản ghi.")
                        st.stop()

                    # --------- Dự đoán trên tập Train ---------
                    data = df['Close'].values.reshape(-1, 1)
                    scaled = scaler.transform(data)
                    if len(scaled) >= window_size:
                        X_all, y_all = create_dataset(scaled, window_size)
                        X_all = X_all.reshape(-1, window_size, 1)
                        train_preds_scaled = model.predict(X_all, verbose=0)
                        train_preds = scaler.inverse_transform(train_preds_scaled).flatten()
                        train_pred_dates = df['Date'].iloc[window_size:window_size + len(train_preds)]



                    # --------- Dự đoán tương lai ---------
                    # Dự đoán tương lai
                    if model_type == "LSTM":
                        future_preds = predict_future(df, scaler, window_size=window_size, steps=steps, selected_symbol=selected_symbol)
                    else:
                        future_preds = predict_future_gru(df, scaler, window_size=window_size, steps=steps, selected_symbol=selected_symbol)


                    # Tạo đúng số lượng ngày giao dịch tiếp theo (bỏ cuối tuần)
                    future_dates = []
                    curr_date = df['Date'].iloc[-1] + pd.Timedelta(days=1)
                    while len(future_dates) < steps:
                        if curr_date.weekday() < 5:  # thứ 2–6
                            future_dates.append(curr_date)
                        curr_date += pd.Timedelta(days=1)

                    # Cắt nếu không khớp độ dài (để tránh lỗi)
                    min_len = min(len(future_dates), len(future_preds))
                    future_dates = future_dates[:min_len]
                    future_preds = future_preds[:min_len]

                    # Tạo bảng dự đoán
                    df_pred = pd.DataFrame({'Ngày': future_dates, 'Giá dự đoán': future_preds})

                    # Đọc dữ liệu thực tế từ thư mục yfinance
                    try:
                        real_path = os.path.join(os.path.dirname(__file__), "yfinance", f"{selected_symbol}.csv")
                        data_real = pd.read_csv(real_path)

                        # Tìm và chuẩn hóa cột ngày
                        for col in ['Date', 'Ngày', 'Unnamed: 0']:
                            if col in data_real.columns:
                                data_real = data_real.rename(columns={col: 'Ngày'})
                                break
                        data_real['Ngày'] = pd.to_datetime(data_real['Ngày'])

                        # Đổi tên cột giá
                        if 'Close' in data_real.columns:
                            data_real = data_real[['Ngày', 'Close']].rename(columns={'Close': 'Giá thực tế'})

                            # Nối và hiển thị bảng
                            merged = pd.merge(df_pred, data_real, on='Ngày', how='left')
                            st.markdown("### DỰ ĐOÁN VÀ THỰC TẾ")
                            st.dataframe(merged, use_container_width=True)

                            # Tính lỗi nếu đủ dữ liệu
                            valid = merged.dropna()
                        else:
                            st.warning("Không có cột 'Close' trong dữ liệu thực tế.")
                    except Exception as e:
                        st.error(f"Lỗi khi đọc dữ liệu thực tế: {e}")


  


                   # ======= Đọc dữ liệu thực tế tương lai từ yfinance =======
                    future_real_path = os.path.join(os.path.dirname(__file__), "yfinance", f"{selected_symbol}.csv")
                    df_real_future = None

                    if os.path.exists(future_real_path):
                        try:
                            df_real_future = pd.read_csv(future_real_path)
                            for col in ['Date', 'Ngày', 'Unnamed: 0']:
                                if col in df_real_future.columns:
                                    df_real_future.rename(columns={col: 'Ngày'}, inplace=True)
                                    break
                            df_real_future['Ngày'] = pd.to_datetime(df_real_future['Ngày'])

                            if 'Close' in df_real_future.columns:
                                df_real_future = df_real_future[['Ngày', 'Close']].rename(columns={'Close': 'Giá thực tế tương lai'})
                        except Exception as e:
                            st.warning(f"⚠️ Không thể đọc giá thực tế tương lai từ yfinance: {e}")


                    # ======= Vẽ biểu đồ =======
                    st.subheader("BIỂU ĐỒ DỰ ĐOÁN")

                    fig = go.Figure()

                    # 1. Giá thực tế lịch sử
                    fig.add_trace(go.Scatter(
                        x=df['Date'], y=df['Close'],
                        mode='lines', name='Giá thực tế',
                        line=dict(color='green')
                    ))

                    # 2. Dự đoán trên tập Train (nếu có)
                    if 'train_preds' in locals() and train_preds is not None:
                        train_pred_dates = df['Date'].iloc[window_size:window_size + len(train_preds)]
                        fig.add_trace(go.Scatter(
                            x=train_pred_dates, y=train_preds,
                            mode='lines', name='Dự đoán (Train)',
                            line=dict(color='orange')
                        ))

                    # 3. Dự đoán tương lai
                    fig.add_trace(go.Scatter(
                        x=df_pred['Ngày'], y=df_pred['Giá dự đoán'],
                        mode='lines', name='Dự đoán tương lai',
                        line=dict(color='red', dash='dot')
                    ))

                    # 4. Giá thực tế tương lai từ yfinance (nếu có)
                    if df_real_future is not None and 'Giá thực tế tương lai' in df_real_future.columns:
                        df_merge_plot = pd.merge(df_pred[['Ngày']], df_real_future, on='Ngày', how='left')
                        if df_merge_plot['Giá thực tế tương lai'].notna().any():
                            fig.add_trace(go.Scatter(
                                x=df_merge_plot['Ngày'], y=df_merge_plot['Giá thực tế tương lai'],
                                mode='lines', name='Giá thực tế tương lai',
                                line=dict(color='blue')
                            ))

                    # Layout chung
                    fig.update_layout(
                        title=f"Dự đoán giá cổ phiếu {selected_symbol} ({model_type})",
                        xaxis_title="Ngày",
                        yaxis_title="Giá cổ phiếu",
                        plot_bgcolor='white',
                        xaxis=dict(showgrid=False),
                        yaxis=dict(showgrid=False),
                        height=600,
                        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                        margin=dict(l=40, r=40, t=80, b=40)
                    )

                    st.plotly_chart(fig, use_container_width=True)

                    # ======= Tính lỗi nếu có dữ liệu thực tế tương lai =======
                    if df_real_future is not None and 'Giá thực tế tương lai' in df_real_future.columns:
                        df_eval = pd.merge(df_pred, df_real_future, on='Ngày', how='inner').dropna()

                        if not df_eval.empty:

                            mae = mean_absolute_error(df_eval['Giá thực tế tương lai'], df_eval['Giá dự đoán'])
                            rmse = np.sqrt(mean_squared_error(df_eval['Giá thực tế tương lai'], df_eval['Giá dự đoán']))
                            mape = np.mean(np.abs((df_eval['Giá thực tế tương lai'] - df_eval['Giá dự đoán']) / df_eval['Giá thực tế tương lai'])) * 100

                            st.markdown("### BẢNG CHỈ SỐ LỖI")
                            df_errors = pd.DataFrame({
                                "MAE": [f"{mae:.2f}"],
                                "RMSE": [f"{rmse:.2f}"],
                                "MAPE (%)": [f"{mape:.2f}"]
                            })
                            st.dataframe(df_errors, use_container_width=True)
                        else:
                            st.info("Không đủ dữ liệu để tính sai số dự đoán tương lai.")

                except Exception as e:
                    import traceback
                    st.error(f"❌ Lỗi khi dự đoán: {e}\n\nChi tiết lỗi:\n{traceback.format_exc()}")







    elif selected_tab == "Optimized ML":
        from tensorflow.keras.models import load_model
        st.markdown("<h1 style='text-align:center;'>OPTIMIZED MACHINE LEARNING MODEL</h1>", unsafe_allow_html=True)
        st.subheader("HUẤN LUYỆN MÔ HÌNH")
        st.subheader("Chọn mô hình huấn luyện")
        model_type = st.selectbox("Loại mô hình:", ["Optimized LSTM", "Optimized GRU"])
        # ======= Chọn loại mô hình nền tảng và có dùng 2 nhánh hay không =======
        model_type_simple = "LSTM" if model_type == "Optimized LSTM" else "GRU"
        branch_model = st.checkbox("Sử dụng mô hình kết hợp (2 nhánh)", value=False)
        # Chọn file dữ liệu
        dataset_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "dataset")
        selected_symbol = st.text_input("Nhập mã chứng khoán", "AAPL").upper().strip()
        selected_file = f"{selected_symbol}.csv"
        file_path = os.path.join(dataset_dir, selected_file)

        # Tham số
        window_size = st.slider("Sliding window", 10, 1000, 200, 5)
        epochs = st.slider("Epochs", 5, 100, 20)
        batch_size = st.slider("Batch size", 8, 512, 64)

        # ===== NÚT HUẤN LUYỆN =====
        if st.button("HUẤN LUYỆN"):
            with st.spinner("Đang huấn luyện mô hình tối ưu, có thể sẽ mất nhiều thời gian của bạn hơn..."):

                # === Thư mục lưu ===
                model_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models_optimal")
                errors_dir = os.path.join(os.path.dirname(model_dir), "errors")
                os.makedirs(model_dir, exist_ok=True)
                os.makedirs(errors_dir, exist_ok=True)

                # === Xác định tên file theo loại mô hình ===
                model_type_simple = "lstm" if "LSTM" in model_type.upper() else "gru"
    
                if branch_model:
                    suffix_model = f"{model_type_simple}_dual_branch_model"
                    suffix_error = f"{model_type_simple}_dual_branch_errors.csv"
                else:
                    suffix_model = f"{model_type_simple}_model"
                    suffix_error = f"{model_type_simple}_errors.csv"

                model_file = f"{selected_symbol}_{suffix_model}.h5"
                model_path = os.path.join(model_dir, model_file)

                error_file = os.path.join(errors_dir, f"{selected_symbol}_{suffix_error}")

                # === Kiểm tra mô hình cũ ===
                is_first_time = not os.path.exists(model_path)
                old_mape = None

                if not is_first_time and os.path.exists(error_file):
                    try:
                        df_errors = pd.read_csv(error_file)
                        if not df_errors.empty:
                            old_mape = df_errors.iloc[-1]['MAPE_Test (%)']
                    except:
                        old_mape = None

                # === HUẤN LUYỆN ===
                try:
                    if branch_model:
                        results = train_dual_branch_model(
                            file_path=file_path,
                            model_type=model_type,
                            window_size=window_size,
                            epochs=epochs,
                            batch_size=batch_size
                        )
                    else:
                        results = train_model_optimized(
                            file_path=file_path,
                            window_size=window_size,
                            epochs=epochs,
                            batch_size=batch_size,
                            model_type=model_type_simple,
                            branch_model=False
                        )
                except Exception as e:
                    st.error(f"Lỗi khi huấn luyện: {e}")
                    st.stop()

                new_mape = results['mape_test']

                # === Thông báo sau huấn luyện ===
                if branch_model:
                    branch_type = "Mô hình **2 nhánh (dual-branch)**"
                else:
                    branch_type = "Mô hình **1 nhánh (single-branch)**"

                st.info(f"""
                ### Thông tin mô hình:
                - Loại mô hình: `{model_type}`
                - Kiểu nhánh: {branch_type}
                - Mã chứng khoán: `{selected_symbol}`
                - Sliding Window: `{window_size}`
                - Epochs: `{epochs}`
                - Batch Size: `{batch_size}`
                """)


                # --- CHUẨN BỊ THƯ MỤC & DANH SÁCH MODEL ---
                base_dir = os.path.dirname(os.path.abspath(__file__))
                dataset_dir = os.path.join(base_dir, "dataset")
                model_dir = os.path.join(base_dir, "models_optimal")
                os.makedirs(dataset_dir, exist_ok=True)
                os.makedirs(model_dir, exist_ok=True)

                # Ở dưới, không cần selectbox nữa, chỉ cần:
                suffix_model = "_lstm_model.h5" if model_type == "Optimized LSTM" else "_gru_model.h5"
                suffix_dual = "_lstm_dual_branch_model.h5" if model_type == "Optimized LSTM" else "_gru_dual_branch_model.h5"

                normal_model_path = os.path.join(model_dir, f"{selected_symbol}{suffix_model}")
                dual_model_path = os.path.join(model_dir, f"{selected_symbol}{suffix_dual}")

                if os.path.exists(dual_model_path):
                    is_dual_branch = True
                elif os.path.exists(normal_model_path):
                    is_dual_branch = False
                else:
                    st.warning("Không tìm thấy model cho mã này.")
                    st.stop()

                # Xác định loại mô hình text để hiển thị
                model_branch_text = "hai nhánh (dual branch)" if branch_model else "một nhánh (single branch)"


                # === Đánh giá cải tiến MAPE ===
                if is_first_time or old_mape is None:
                    st.success(
                        f"Đây là mô hình đầu tiên cho mã `{selected_symbol}`.\n"
                        f"Loại mô hình: **{model_type} – {model_branch_text}**. Đã lưu thành công!"
                    )
                else:
                    if new_mape < old_mape:
                        st.success(
                            f"Mô hình mới (**{model_type} – {model_branch_text}**) tốt hơn mô hình cũ "
                            f"(MAPE Test giảm từ `{old_mape:.2f}%` xuống `{new_mape:.2f}%`). Đã cập nhật!"
                        )
                    elif new_mape > old_mape:
                        st.warning(
                            f"Mô hình mới (**{model_type} – {model_branch_text}**) có MAPE Test `{new_mape:.2f}%` "
                            f"cao hơn mô hình cũ `{old_mape:.2f}%`. Đang giữ mô hình cũ."
                        )
                    else:
                        st.info(
                            f"Mô hình mới (**{model_type} – {model_branch_text}**) có MAPE Test bằng mô hình cũ "
                            f"({new_mape:.2f}%)."
                        )


                


                st.subheader("BIỂU ĐỒ TRAIN & TEST")
                # --- ĐỌC METADATA LẤY window_size ĐÚNG CỦA MODEL ---
                meta_ext = (
                    f"{model_type.lower().split()[-1]}_dual_branch_metadata.json"
                    if is_dual_branch else
                    f"{model_type.lower().split()[-1]}_metadata.json"
                )
                metadata_path = os.path.join(model_dir, f"{selected_symbol}_{meta_ext}")
                window_size = 60  # fallback default



                # 1. Xác định file model & metadata
                model_type_simple = "lstm" if "LSTM" in model_type.upper() else "gru"
                if is_dual_branch:
                    model_suffix = f"_{model_type_simple}_dual_branch_model.h5"
                    metadata_name = f"{selected_symbol}_{model_type_simple}_dual_branch_metadata.json"
                else:
                    model_suffix = f"_{model_type_simple}_model.h5"
                    metadata_name = f"{selected_symbol}_{model_type_simple}_metadata.json"

                model_path = os.path.join(model_dir, f"{selected_symbol}{model_suffix}")
                metadata_path = os.path.join(model_dir, metadata_name)

                # --- **LUÔN LOAD MODEL TRƯỚC** ---
                model = load_model(model_path)   # <- PHẢI ĐỨNG TRƯỚC!

                # --- SAU ĐÓ mới lấy window_size ---
                if os.path.exists(metadata_path):
                    with open(metadata_path, "r") as f:
                        metadata = json.load(f)
                    window_size = metadata.get("window_size", model.input_shape[1])  # Được phép dùng model.input_shape[1] vì đã load rồi
                else:
                    window_size = model.input_shape[1]


                # 1. Đọc dữ liệu
                csv_path = os.path.join("dataset", f"{selected_symbol}.csv")
                if not os.path.exists(csv_path):
                    st.error(f"❌ Không tìm thấy file dữ liệu: {csv_path}")
                    st.stop()
                df = pd.read_csv(csv_path)
                df['Date'] = pd.to_datetime(df['Date'])
                df = df.sort_values('Date')

                # 1. Load scaler trước!
                if is_dual_branch:
                    scaler_close_name = f"{selected_symbol}_{model_type_simple}_dual_scaler_close.pkl"
                    scaler_multi_name = f"{selected_symbol}_{model_type_simple}_dual_scaler_multi.pkl"
                    scaler_close_path = os.path.join("models_optimal", scaler_close_name)
                    scaler_multi_path = os.path.join("models_optimal", scaler_multi_name)
                    if not os.path.exists(scaler_close_path) or not os.path.exists(scaler_multi_path):
                        st.error(f"❌ Không tìm thấy scaler dual cho mô hình: {scaler_close_path} hoặc {scaler_multi_path}")
                        st.stop()
                    scaler_close = joblib.load(scaler_close_path)
                    scaler_multi = joblib.load(scaler_multi_path)
                else:
                    scaler_name = f"{selected_symbol}_{model_type_simple}_scaler.pkl"
                    scaler_path = os.path.join("models_optimal", scaler_name)
                    if not os.path.exists(scaler_path):
                        st.error(f"❌ Không tìm thấy scaler cho mô hình: {scaler_path}")
                        st.stop()
                    scaler = joblib.load(scaler_path)

                if is_dual_branch:
                    df_feature = add_technical_indicators(df.copy())
                    df_feature = df_feature.dropna().reset_index(drop=True)
                    feature_cols = [
                        'close_lag1',
                        'close_lag5',
                        'close_lag10',
                        'log_return'
                    ]
                    multi_data = df_feature[feature_cols].values
                    close_data = df_feature['Close'].values.reshape(-1, 1)


                    scaled_multi = scaler_multi.transform(multi_data)
                    scaled_close = scaler_close.transform(close_data)

                    if len(scaled_multi) >= window_size and len(scaled_close) >= window_size:
                        X_multi_all = np.array([scaled_multi[i:i + window_size] for i in range(len(scaled_multi) - window_size)])
                        X_close_all = np.array([scaled_close[i:i + window_size] for i in range(len(scaled_close) - window_size)])
                        X_close_all = X_close_all.reshape(-1, window_size, 1)

                        model_suffix = f"{model_type_simple}_dual_branch_model"
                        model_path = os.path.join(model_dir, f"{selected_symbol}_{model_suffix}.h5")
                        model = load_model(model_path)
                        train_preds_scaled = model.predict([X_multi_all, X_close_all], verbose=0)
                        train_preds = scaler_close.inverse_transform(train_preds_scaled).flatten()
                        # Lấy Date từ df_feature (vì đã dropna đầu chuỗi!)
                        train_pred_dates = df_feature['Date'].iloc[window_size:window_size + len(train_preds)]
                    else:
                        train_preds = None
                        train_pred_dates = None

                else:
                    df_close_only = df[['Date', 'Close']].copy()
                    data = df_close_only['Close'].values.reshape(-1, 1)
                    scaled_data = scaler.transform(data)
                    if len(scaled_data) >= window_size:
                        X_all = np.array([scaled_data[i:i + window_size] for i in range(len(scaled_data) - window_size)])
                        X_all = X_all.reshape(-1, window_size, 1)
                        model_suffix = f"{model_type_simple}_model"
                        model_path = os.path.join(model_dir, f"{selected_symbol}_{model_suffix}.h5")
                        model = load_model(model_path)
                        train_preds_scaled = model.predict(X_all, verbose=0)
                        train_preds = scaler.inverse_transform(train_preds_scaled).flatten()
                        train_pred_dates = df_close_only['Date'].iloc[window_size:window_size + len(train_preds)]
                    else:
                        train_preds = None
                        train_pred_dates = None



                fig = go.Figure()
                # Vẽ tập Train theo thời gian
                fig.add_trace(go.Scatter(
                    x=results['train_dates'], y=results['y_train_real'].flatten(),
                    name="Giá thực tế (Train)", line=dict(color="blue")
                ))
                fig.add_trace(go.Scatter(
                    x=results['train_dates'], y=results['y_train_pred'].flatten(),
                    name="Dự đoán (Train)", line=dict(color="orange")
                ))

                # Vẽ tập Test theo thời gian
                fig.add_trace(go.Scatter(
                    x=results['test_dates'], y=results['y_test_real'].flatten(),
                    name="Giá thực tế (Test)", line=dict(color="green"), mode="lines"
                ))
                fig.add_trace(go.Scatter(
                    x=results['test_dates'], y=results['y_test_pred'].flatten(),
                    name="Dự đoán (Test)", line=dict(color="red"), mode="lines"
                ))
                # Tùy chỉnh layout
                fig.update_layout(
                    title=f"Dự đoán giá cổ phiếu {selected_symbol} ({model_type})",
                    xaxis_title="Ngày",
                    yaxis_title="Giá trị",
                    plot_bgcolor="white",
                    xaxis=dict(showgrid=False),
                    yaxis=dict(showgrid=False),
                    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                    margin=dict(l=40, r=40, t=80, b=40)
                )
                st.plotly_chart(fig, use_container_width=True)

                st.markdown("### ĐÁNH GIÁ MÔ HÌNH")

                col1, col2 = st.columns(2)

                with col1:
                    st.markdown("Train")
                    st.metric("MAE", f"{results['mae_train']:.2f}")
                    st.metric("RMSE", f"{results['rmse_train']:.2f}")
                    st.metric("MAPE(%)", f"{results['mape_train']:.2f}") 

                with col2:
                    st.markdown("Test")
                    st.metric("MAE", f"{results['mae_test']:.2f}")  
                    st.metric("RMSE", f"{results['rmse_test']:.2f}")
                    st.metric("MAPE(%)", f"{results['mape_test']:.2f}")

                st.markdown("---")
                st.markdown("### SO SÁNH GIÁ TRAIN & TEST VỚI THỰC TẾ (15 dòng gần nhất)")

                col1, col2 = st.columns(2)

                with col1:
                    st.markdown("Train")
                    df_train = pd.DataFrame({
                        "Ngày": results['train_dates'].reset_index(drop=True),
                        "Giá thực tế": results['y_train_real'].flatten(),
                        "Dự đoán": results['y_train_pred'].flatten()
                    })
                    st.dataframe(df_train.tail(15), use_container_width=True)

                with col2:
                    st.markdown("Test")
                    df_test = pd.DataFrame({
                        "Ngày": results['test_dates'].reset_index(drop=True),
                        "Giá thực tế": results['y_test_real'].flatten(),
                        "Dự đoán": results['y_test_pred'].flatten()
                    }) 

                    st.dataframe(df_test.tail(15), use_container_width=True)


        # ==== DỰ ĐOÁN TỪ MÔ HÌNH ====
        st.title("DỰ ĐOÁN TỪ MÔ HÌNH")

        # 1. Setup thư mục
        base_dir = os.path.dirname(os.path.abspath(__file__))
        dataset_dir = os.path.join(base_dir, "dataset")
        model_dir = os.path.join(base_dir, "models_optimal")
        os.makedirs(dataset_dir, exist_ok=True)
        os.makedirs(model_dir, exist_ok=True)

        # 2. Lấy danh sách model (single/dual, lstm/gru)
        suffix_model = "_lstm_model.h5" if model_type == "Optimized LSTM" else "_gru_model.h5"
        suffix_dual = "_lstm_dual_branch_model.h5" if model_type == "Optimized LSTM" else "_gru_dual_branch_model.h5"

        all_model_files = os.listdir(model_dir)
        normal_models = [f for f in all_model_files if f.endswith(suffix_model)]
        dual_models = [f for f in all_model_files if f.endswith(suffix_dual)]

        model_map = {}
        display_models = []
        for f in normal_models:
            symbol = f.replace(suffix_model, "")
            label = f"{symbol} (Single)"
            display_models.append(label)
            model_map[label] = {"symbol": symbol, "is_dual": False}
        for f in dual_models:
            symbol = f.replace(suffix_dual, "")
            label = f"{symbol} (Dual)"
            display_models.append(label)
            model_map[label] = {"symbol": symbol, "is_dual": True}
        if not display_models:
            st.warning(f"Không có mô hình {model_type} nào đã huấn luyện.")
            st.stop()

        # 3. Chọn mã CK & kiểu model
        selected_display = st.selectbox("Chọn mã chứng khoán ", display_models)
        selected_symbol = model_map[selected_display]["symbol"]
        is_dual_branch = model_map[selected_display]["is_dual"]

        # 1. Xác định tên file model & metadata dựa theo is_dual_branch
        # 1. Xác định đường dẫn model & metadata
        # Đầu tiên: xác định file model và metadata
        model_type_simple = "lstm" if "LSTM" in model_type.upper() else "gru"
        if is_dual_branch:
            model_suffix = f"_{model_type_simple}_dual_branch_model.h5"
            metadata_name = f"{selected_symbol}_{model_type_simple}_dual_branch_metadata.json"
        else:
            model_suffix = f"_{model_type_simple}_model.h5"
            metadata_name = f"{selected_symbol}_{model_type_simple}_metadata.json"

        model_path = os.path.join(model_dir, f"{selected_symbol}{model_suffix}")
        metadata_path = os.path.join(model_dir, metadata_name)

        # TIẾP THEO: LOAD MODEL Ở ĐÂY!
        model = load_model(model_path)   # <- DÒNG NÀY LUÔN PHẢI ĐỨNG TRƯỚC!

        # SAU ĐÓ mới lấy window_size
        if os.path.exists(metadata_path):
            with open(metadata_path, "r") as f:
                metadata = json.load(f)
            window_size = metadata.get("window_size", model.input_shape[1])  # Lúc này model đã có rồi
        else:
            window_size = model.input_shape[1]







        # 6. Load scaler
        if is_dual_branch:
            scaler_close_name = f"{selected_symbol}_{model_type_simple}_dual_scaler_close.pkl"
            scaler_multi_name = f"{selected_symbol}_{model_type_simple}_dual_scaler_multi.pkl"
            scaler_close_path = os.path.join(model_dir, scaler_close_name)
            scaler_multi_path = os.path.join(model_dir, scaler_multi_name)
            if not os.path.exists(scaler_close_path) or not os.path.exists(scaler_multi_path):
                st.error(f"❌ Không tìm thấy scaler dual cho mô hình: {scaler_close_path} hoặc {scaler_multi_path}")
                st.stop()
            scaler_close = joblib.load(scaler_close_path)
            scaler_multi = joblib.load(scaler_multi_path)
        else:
            scaler_name = f"{selected_symbol}_{model_type_simple}_scaler.pkl"
            scaler_path = os.path.join(model_dir, scaler_name)
            if not os.path.exists(scaler_path):
                st.error(f"❌ Không tìm thấy scaler cho mô hình: {scaler_path}")
                st.stop()
            scaler = joblib.load(scaler_path)

        # 7. Load dataset
        dataset_path = os.path.join(dataset_dir, f"{selected_symbol}.csv")
        if not os.path.exists(dataset_path):
            st.error(f"Không tìm thấy dữ liệu: `{dataset_path}`.")
            st.stop()

        df = pd.read_csv(dataset_path)
        df['Date'] = pd.to_datetime(df['Date'])
        df = df.sort_values('Date')
        if len(df) < window_size:
            st.error(f"Dữ liệu không đủ để dự đoán. Cần {window_size} bản ghi.")
            st.stop()

        steps = st.number_input("Số ngày muốn dự đoán", min_value=1, max_value=365, value=22, step=1)

        df_real_future = None
        df_pred = None

        if st.button("DỰ ĐOÁN"):
            with st.spinner('Đang dự đoán, vui lòng chờ...'):
                try:



                    # ==== 1. Dự đoán tương lai ====
                    future_preds = predict_goodnine(
                        df=df,
                        scaler=scaler_close if is_dual_branch else scaler,
                        window_size=window_size,
                        steps=steps,
                        selected_symbol=selected_symbol,
                        model_type=model_type,
                        is_dual_branch=is_dual_branch
                    )

                    # 3. KIỂM TRA GIÁ TRỊ DỰ ĐOÁN ĐẦU TIÊN
                    st.write("Giá trị đầu tiên dự đoán tương lai:", future_preds[0])
                    st.write("Giá cuối cùng của thực tế:", df['Close'].iloc[-1])

                    future_dates = []
                    # Ngày bắt đầu dự báo là ngày cuối cùng trong dataset + 1
                    future_dates = []
                    curr_date = df['Date'].max() + pd.Timedelta(days=1)
                    while len(future_dates) < steps:
                        if curr_date.weekday() < 5:  # Chỉ thứ 2-6
                            future_dates.append(curr_date)
                        curr_date += pd.Timedelta(days=1)
                    df_pred = pd.DataFrame({"Ngày": future_dates, "Giá dự đoán": future_preds})


                    train_preds = None
                    train_pred_dates = None
                    if is_dual_branch:
                        df_feature = add_technical_indicators(df.copy())
                        df_feature = df_feature.dropna().reset_index(drop=True)
                        feature_cols = [
                            'close_lag1',
                            'close_lag5',
                            'close_lag10',
                            'log_return'
                        ]
                        multi_data = df_feature[feature_cols].values
                        close_data = df_feature['Close'].values.reshape(-1, 1)
                        scaled_multi = scaler_multi.transform(multi_data)
                        scaled_close = scaler_close.transform(close_data)
                        if len(scaled_multi) >= window_size and len(scaled_close) >= window_size:
                            X_multi_all, y_all = create_dataset_multi(scaled_multi, scaled_close, window_size)
                            X_close_all, _ = create_dataset_multi(scaled_close, scaled_close, window_size)
                            X_close_all = X_close_all.reshape(-1, window_size, 1)
                            train_preds_scaled = model.predict([X_multi_all, X_close_all], verbose=0)
                            train_preds = scaler_close.inverse_transform(train_preds_scaled).flatten()
                            train_pred_dates = df_feature['Date'].iloc[window_size:window_size + len(train_preds)]
                    else:
                        data = df['Close'].values.reshape(-1, 1)
                        scaled = scaler.transform(data)
                        if len(scaled) >= window_size:
                               X_all, y_all = create_dataset_uni(scaled, window_size)
                        X_all = X_all.reshape(-1, window_size, 1)
                        train_preds_scaled = model.predict(X_all, verbose=0)
                        train_preds = scaler.inverse_transform(train_preds_scaled).flatten()
                        train_pred_dates = df['Date'].iloc[window_size:window_size + len(train_preds)]


                    # ==== 3. Đọc dữ liệu thực tế tương lai (nếu có) ====
                    df_real_future = None
                    future_real_path = os.path.join(base_dir, "yfinance", f"{selected_symbol}.csv")
                    if os.path.exists(future_real_path):
                        df_real_future = pd.read_csv(future_real_path)
                        for col in ['Date', 'Ngày', 'Unnamed: 0']:
                            if col in df_real_future.columns:
                                df_real_future.rename(columns={col: 'Ngày'}, inplace=True)
                                break
                        df_real_future['Ngày'] = pd.to_datetime(df_real_future['Ngày'])
                        if 'Close' in df_real_future.columns:
                            df_real_future = df_real_future[['Ngày', 'Close']].rename(columns={'Close': 'Giá thực tế'})

                    # ==== 4. Hiển thị bảng dự đoán ====
                    if df_real_future is not None and 'Giá thực tế' in df_real_future.columns:
                        df_eval = pd.merge(df_pred, df_real_future, on='Ngày', how='left')
                        st.markdown("### BẢNG DỰ ĐOÁN VÀ GIÁ THỰC TẾ")
                        st.dataframe(df_eval, use_container_width=True)
                    else:
                        st.markdown("### BẢNG DỰ ĐOÁN")
                        st.dataframe(df_pred, use_container_width=True)

                    # --- LẤY DỰ ĐOÁN TOÀN BỘ NHANH ---
                    if is_dual_branch:
                        X_multi_all = np.array([scaled_multi[i:i + window_size] for i in range(len(scaled_multi) - window_size)])
                        X_close_all = np.array([scaled_close[i:i + window_size] for i in range(len(scaled_close) - window_size)])
                        X_close_all = X_close_all.reshape(-1, window_size, 1)
                        full_preds_scaled = model.predict([X_multi_all, X_close_all], verbose=0)
                        full_preds = scaler_close.inverse_transform(full_preds_scaled).flatten()
                        full_pred_dates = df_feature['Date'].iloc[window_size:window_size + len(full_preds)]
                    else:
                        X_all = np.array([scaled[i:i + window_size] for i in range(len(scaled) - window_size)])
                        X_all = X_all.reshape(-1, window_size, 1)
                        full_preds_scaled = model.predict(X_all, verbose=0)
                        full_preds = scaler.inverse_transform(full_preds_scaled).flatten()
                        full_pred_dates = df['Date'].iloc[window_size:window_size + len(full_preds)]




                    def safe_list(x):
                        # Chuyển Series, ndarray thành list; nếu None thì trả về []
                        if isinstance(x, (pd.Series, np.ndarray)):
                            return list(x)
                        if x is None:
                            return []
                        return x

                    # Chuẩn hóa dữ liệu đầu vào cho chart
                    train_pred_dates = safe_list(train_pred_dates)
                    train_preds = safe_list(train_preds)
                    future_dates = safe_list(future_dates)
                    future_preds = safe_list(future_preds)
                    full_pred_dates = safe_list(full_pred_dates)
                    full_preds = safe_list(full_preds)

                    
                    # Biểu đồ
                    st.subheader("BIỂU ĐỒ DỰ ĐOÁN")
                    fig = go.Figure()

                    # Giá thực tế
                    fig.add_trace(go.Scatter(
                        x=df['Date'], y=df['Close'],
                        mode='lines', name='Giá thực tế', line=dict(color='green')
                    ))

                    # 2. Đường dự đoán trên tập train (màu cam)
                    fig.add_trace(go.Scatter(
                        x=train_pred_dates, y=train_preds,
                        mode='lines', name='Dự đoán (Train)', line=dict(color='orange', dash='dot')
                    ))

                    # Đường dự đoán tương lai, bắt đầu từ ngày tiếp theo
                    fig.add_trace(go.Scatter(
                        x=future_dates, y=future_preds,
                        mode='lines', name='Dự đoán tương lai', line=dict(color='red', dash='dot')
                    ))

                    # Giá thực tế tương lai (nếu có, có thể thêm)
                    if df_real_future is not None and 'Giá thực tế' in df_real_future.columns:
                        df_merge_plot = pd.merge(pd.DataFrame({'Ngày': future_dates}), df_real_future, on='Ngày', how='left')
                        if df_merge_plot['Giá thực tế'].notna().any():
                            fig.add_trace(go.Scatter(
                                x=df_merge_plot['Ngày'], y=df_merge_plot['Giá thực tế'],
                                mode='lines', name='Giá thực tế tương lai', line=dict(color='blue')
                            ))

                    # Layout đẹp
                    fig.update_layout(
                        title=f"Dự đoán giá cổ phiếu {selected_symbol} ({model_type})",
                        xaxis_title="Ngày", yaxis_title="Giá cổ phiếu",
                        plot_bgcolor='white',
                        xaxis=dict(showgrid=False), yaxis=dict(showgrid=False),
                        height=600,
                        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                        margin=dict(l=40, r=40, t=80, b=40)
                    )
                    st.plotly_chart(fig, use_container_width=True)

                    # ==== 6. Tính và hiển thị bảng chỉ số lỗi ====
                    if df_real_future is not None and 'Giá thực tế' in df_real_future.columns:
                        df_eval = pd.merge(df_pred, df_real_future, on='Ngày', how='inner').dropna()
                        if not df_eval.empty:
                            mae = mean_absolute_error(df_eval['Giá thực tế'], df_eval['Giá dự đoán'])
                            rmse = np.sqrt(mean_squared_error(df_eval['Giá thực tế'], df_eval['Giá dự đoán']))
                            mape = np.mean(np.abs((df_eval['Giá thực tế'] - df_eval['Giá dự đoán']) / df_eval['Giá thực tế'])) * 100

                            st.markdown("### BẢNG CHỈ SỐ LỖI")
                            df_errors = pd.DataFrame({
                                "MAE": [f"{mae:.2f}"],
                                "RMSE": [f"{rmse:.2f}"],
                                "MAPE (%)": [f"{mape:.2f}"]
                            })
                            st.dataframe(df_errors, use_container_width=True)
                        else:
                            st.info("Không đủ dữ liệu để tính sai số dự đoán tương lai.")

                except Exception as e:
                    st.error(f"\u274c Lỗi khi dự đoán: {e}")




    if selected_tab == "Model Performance":
        st.markdown("<h1 style='text-align: center;'>SO SÁNH HIỆU QUẢ CÁC MÔ HÌNH</h1>", unsafe_allow_html=True)

        errors_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "errors")

        if os.path.exists(errors_dir):
            error_files = [f for f in os.listdir(errors_dir) if f.endswith('.csv')]

            if error_files:
                all_errors = []

                for error_file in error_files:
                    error_path = os.path.join(errors_dir, error_file)
                    df_error = pd.read_csv(error_path)
                    all_errors.append(df_error)

                df_all_errors = pd.concat(all_errors, ignore_index=True)

                # Nút hiển thị và tải xuống
                st.subheader("Tổng Hợp Chỉ Số Lỗi")
                st.dataframe(df_all_errors, use_container_width=True)

                st.download_button("Tải Bảng Lỗi Tổng Hợp", df_all_errors.to_csv(index=False).encode('utf-8'),
                                file_name="all_model_errors.csv", mime="text/csv")

            else:
                st.info("Thư mục 'errors' chưa có file lỗi nào.")
        else:
            st.warning("Không tìm thấy thư mục 'errors'.")

        st.markdown("""
            <div style='text-align: center; margin-top: 40px; font-size: 20px; color: gray;'>
                <em>Tính năng đang được phát triển – Coming Soon</em>
            </div>str
        """, unsafe_allow_html=True)




if __name__ == "__main__":
    main()



