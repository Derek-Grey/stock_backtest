# -*- coding: utf-8 -*-
# title: 快速回测
"""
快速回测页面
允许用户上传权重和收益率CSV文件进行回测
"""
import sys
from pathlib import Path
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime
from loguru import logger
import sqlite3
import pymongo

# 添加项目根目录到系统路径
ROOT_DIR = Path(__file__).resolve().parent.parent.parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

from src.backtest.portfolio_metrics import DataChecker, calculate_portfolio_metrics
from config.settings import DATA_DIR, OUTPUT_DIR

def validate_weight_csv(df):
    """验证权重CSV文件格式"""
    required_columns = ['date', 'code', 'weight']
    
    # 检查必需列是否存在
    if not all(col in df.columns for col in required_columns):
        missing_cols = [col for col in required_columns if col not in df.columns]
        return False, f"权重CSV文件缺少必需列: {', '.join(missing_cols)}"
    
    # 检查日期格式
    try:
        pd.to_datetime(df['date'])
    except:
        return False, "日期列格式不正确，应为YYYY-MM-DD格式"
    
    return True, "验证通过"

def validate_return_csv(df):
    """验证收益率CSV文件格式"""
    required_columns = ['date', 'code', 'return']
    
    # 检查必需列是否存在
    if not all(col in df.columns for col in required_columns):
        missing_cols = [col for col in required_columns if col not in df.columns]
        return False, f"收益率CSV文件缺少必需列: {', '.join(missing_cols)}"
    
    # 检查日期格式
    try:
        pd.to_datetime(df['date'])
    except:
        return False, "日期列格式不正确，应为YYYY-MM-DD格式"
    
    return True, "验证通过"

def get_db_connection():
    """创建到SQLite数据库的连接。"""
    conn = sqlite3.connect('backtest_results.db')
    return conn

def create_results_table():
    """如果不存在，则创建用于存储回测结果的表。"""
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS backtest_results (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            total_return REAL,
            annual_return REAL,
            volatility REAL,
            sharpe REAL,
            max_drawdown REAL,
            avg_turnover REAL,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    conn.commit()
    conn.close()

def quick_backtest_page():
    st.title("📈 快速回测")
    
    st.markdown(
        """
        <style>
        .main-title {
            font-size: 2rem;  /* Unified font size */
        }
        </style>
        """, unsafe_allow_html=True
    )
    
    # 在 quick_backtest_page 函数中添加用户自定义选项
    st.sidebar.header("图表设置")
    color_option = st.sidebar.color_picker("选择图表颜色", "#1f77b4")

    # CSV Template Downloads
    st.markdown("### 下载CSV模板")
    st.download_button(
        label="下载权重CSV模板",
        data="date,code,weight\n2023-01-03,SH600788,0.1\n2023-01-03,SZ000765,0.2\n",
        file_name="weight_template.csv",
        mime="text/csv"
    )
    
    st.download_button(
        label="下载收益率CSV模板",
        data="date,code,return\n2023-01-03,SH600000,-0.00688\n2023-01-03,SZ000765,-0.00233\n",
        file_name="return_template.csv",
        mime="text/csv"
    )

    # 文件上传
    col1, col2 = st.columns(2)
    with col1:
        weight_file = st.file_uploader("上传权重矩阵CSV文件", type=['csv'])
    with col2:
        return_file = st.file_uploader("上传收益率矩阵CSV文件", type=['csv'])
    
    if weight_file is not None and return_file is not None:
        try:
            # 读取CSV文件
            weight_df = pd.read_csv(weight_file)
            return_df = pd.read_csv(return_file)
            
            # 验证文件格式
            is_valid_weight, weight_message = validate_weight_csv(weight_df)
            is_valid_return, return_message = validate_return_csv(return_df)
            
            if not is_valid_weight:
                st.error(weight_message)
                return
            if not is_valid_return:
                st.error(return_message)
                return
                
            # 保存上传的文件
            weight_path = Path(DATA_DIR) / 'test_weight.csv'
            return_path = Path(DATA_DIR) / 'test_return.csv'
            weight_df.to_csv(weight_path, index=False)
            return_df.to_csv(return_path, index=False)
            
            st.success("文件上传成功！")
            
            # 添加确认按钮
            if st.button("开始回测"):
                # 检查数据格式
                checker = DataChecker()
                try:
                    checker.check_trading_dates(weight_df)
                    checker.check_trading_dates(return_df)
                except ValueError as e:
                    st.error(f"数据验证失败: {str(e)}")
                    return
                
                # 执行回测
                try:
                    with st.spinner("正在执行回测..."):
                        portfolio_returns, turnover = calculate_portfolio_metrics(
                            str(weight_path),
                            str(return_path)
                        )
                        
                        # 显示回测结果
                        st.subheader("回测结果")
                        display_metrics(portfolio_returns, turnover)
                        
                        # 添加收益分布直方图
                        hist_fig = plot_return_distribution(portfolio_returns, color_option)
                        st.plotly_chart(hist_fig, use_container_width=True)
                        
                        # 显示策略表现图
                        fig = plot_cumulative_returns(portfolio_returns)
                        if fig:
                            st.plotly_chart(fig, use_container_width=True)
                        
                except Exception as e:
                    st.error(f"回测执行失败: {str(e)}")
                    logger.exception("回测失败详细信息:")
                
        except Exception as e:
            st.error(f"文件处理失败: {str(e)}")

def display_metrics(returns, turnover):
    """显示策略表现指标"""
    col1, col2, col3 = st.columns(3)
    
    # 计算关键指标
    total_return = (1 + returns).prod() - 1
    is_intraday = isinstance(returns.index[0], pd.Timestamp) and returns.index[0].hour != 0
    annual_factor = 252 if not is_intraday else 252 * 240
    
    annual_return = (1 + total_return) ** (annual_factor / len(returns)) - 1
    volatility = returns.std() * (annual_factor ** 0.5)
    sharpe = annual_return / volatility if volatility != 0 else 0
    max_drawdown = ((1 + returns).cumprod() / 
                    (1 + returns).cumprod().cummax() - 1).min()
    avg_turnover = turnover.mean()
    
    with col1:
        st.metric("总收益率", f"{total_return:.2%}")
        st.metric("年化收益率", f"{annual_return:.2%}")
    
    with col2:
        st.metric("年化波动率", f"{volatility:.2%}")
        st.metric("夏普比率", f"{sharpe:.2f}")
    
    with col3:
        st.metric("最大回撤", f"{max_drawdown:.2%}")
        st.metric("平均换手率", f"{avg_turnover:.2%}")

    # 准备要保存到MongoDB的数据
    result_data = {
        "total_return": total_return,
        "annual_return": annual_return,
        "volatility": volatility,
        "sharpe": sharpe,
        "max_drawdown": max_drawdown,
        "avg_turnover": avg_turnover,
        "timestamp": datetime.now()
    }

    # 将结果保存到MongoDB
    save_to_mongo(result_data, "quick_backtest_results")

def plot_cumulative_returns(returns):
    """绘制累计收益率和回撤图表"""
    # 确保数据按时间排序
    returns = returns.sort_index()
    
    # 计算累计收益率和回撤
    cumulative_returns = (1 + returns).cumprod() - 1
    rolling_max = (1 + returns).cumprod().cummax()
    drawdowns = (1 + returns).cumprod() / rolling_max - 1
    
    # 获取有效的时间点并格式化
    valid_times = returns[returns.notna()].index
    formatted_times = [t.strftime('%Y-%m-%d %H:%M:%S') if isinstance(t, pd.Timestamp) and t.hour != 0 
                      else t.strftime('%Y-%m-%d') 
                      for t in valid_times]
    
    # 创建图表
    fig = go.Figure()
    
    # 定义颜色
    profit_color = '#1f77b4'  # 蓝色用于收益率
    drawdown_color = '#ff7f0e'  # 橙色用于回撤
    
    # 添加累计收益率曲线
    fig.add_trace(go.Scatter(
        x=formatted_times,
        y=cumulative_returns[valid_times].values,
        mode='lines',
        name='累计收益率',
        yaxis='y1',
        line=dict(color=profit_color, width=3),
        hovertemplate='%{x}<br>收益率: %{y:.2%}<extra></extra>'
    ))
    
    # 添加回撤曲线（使用填充区域）
    fig.add_trace(go.Scatter(
        x=formatted_times,
        y=drawdowns[valid_times].values,
        mode='none',
        name='回撤',
        yaxis='y2',
        fill='tozeroy',
        fillcolor='rgba(255, 127, 14, 0.3)',  # 半透明的橙色
        hovertemplate='%{x}<br>回撤: %{y:.2%}<extra></extra>'
    ))
    
    # 更新布局
    fig.update_layout(
        title='策略表现',
        plot_bgcolor='white',  # 设置为白色
        paper_bgcolor='white',  # 设置为白色
        yaxis=dict(
            title='累计收益率',
            title_font=dict(color=profit_color),
            tickfont=dict(color=profit_color),
            tickformat='.2%',
            gridcolor='lightgrey',
            showgrid=True,
            zeroline=True,
            zerolinecolor='lightgrey'
        ),
        yaxis2=dict(
            title='回撤',
            title_font=dict(color=drawdown_color),
            tickfont=dict(color=drawdown_color),
            overlaying='y',
            side='right',
            tickformat='.2%',
            gridcolor='lightgrey',
            showgrid=False,
            range=[min(drawdowns[valid_times].values) * 1.1, 0]
        ),
        xaxis=dict(
            type='category',
            showgrid=True,
            gridcolor='lightgrey',
            tickfont=dict(size=10),
            tickangle=45,
            tickmode='array',
            ticktext=formatted_times,
            tickvals=formatted_times,
            showticklabels=True  # 显示刻度标签
        ),
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=0.01
        ),
        hoverlabel=dict(
            bgcolor="white",
            font_size=12,
            font_family="Arial"
        )
    )
    
    return fig

def plot_return_distribution(returns, color_option):
    """绘制收益分布的直方图"""
    fig = go.Figure()
    fig.add_trace(go.Histogram(
        x=returns,
        nbinsx=50,
        marker_color=color_option  # 使用用户选择的颜色
    ))
    
    fig.update_layout(
        title='收益分布',
        xaxis_title='收益率',
        yaxis_title='频率',
        bargap=0.2
    )
    
    return fig

def save_to_mongo(data, collection_name):
    """将结果保存到本地MongoDB数据库。"""
    try:
        # 连接到本地MongoDB服务器
        client = pymongo.MongoClient("mongodb://localhost:27017/")
        # 选择数据库
        db = client["backtest_results"]
        # 选择集合
        collection = db[collection_name]
        # 插入数据
        collection.insert_one(data)
        st.success(f"结果已保存到本地数据库: {collection_name}")
    except Exception as e:
        st.error(f"无法保存到数据库: {str(e)}")
        logger.exception("保存到数据库失败详细信息:")

if __name__ == "__main__":
    # 在脚本开始时调用此函数以确保表已创建
    create_results_table()
    quick_backtest_page() 