"""
回测策略的 Streamlit 前端界面
"""
import sys
from pathlib import Path
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime

# 添加项目根目录到系统路径
ROOT_DIR = Path(__file__).resolve().parent.parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

from src.backtest.main import main

def plot_cumulative_returns(results, strategy_name):
    """绘制累计收益率图表"""
    if results is None:
        return None
        
    cumulative_returns = (1 + results['daily_return']).cumprod() - 1
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=cumulative_returns.index,
        y=cumulative_returns.values,
        mode='lines',
        name=f'{strategy_name}策略'
    ))
    
    fig.update_layout(
        title=f'{strategy_name}策略累计收益率',
        xaxis_title='日期',
        yaxis_title='累计收益率',
        yaxis_tickformat='.2%'
    )
    
    return fig

def display_metrics(results, strategy_name):
    """显示策略表现指标"""
    if results is None:
        st.warning(f"{strategy_name}策略没有产生结果")
        return
        
    daily_returns = results['daily_return']
    
    # 计算关键指标
    total_return = (1 + daily_returns).cumprod().iloc[-1] - 1
    annual_return = (1 + total_return) ** (252 / len(daily_returns)) - 1
    volatility = daily_returns.std() * (252 ** 0.5)
    sharpe = annual_return / volatility if volatility != 0 else 0
    max_drawdown = ((1 + daily_returns).cumprod() / (1 + daily_returns).cumprod().cummax() - 1).min()
    
    # 创建指标展示列
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("总收益率", f"{total_return:.2%}")
        st.metric("年化收益率", f"{annual_return:.2%}")
    
    with col2:
        st.metric("年化波动率", f"{volatility:.2%}")
        st.metric("夏普比率", f"{sharpe:.2f}")
    
    with col3:
        st.metric("最大回撤", f"{max_drawdown:.2%}")

def main_app():
    """主应用程序"""
    st.set_page_config(page_title="量化策略回测系统", layout="wide")
    
    st.title("量化策略回测系统")
    
    # 侧边栏 - 参数设置
    st.sidebar.header("参数设置")
    
    # 日期选择
    min_date = datetime(2010, 1, 1)
    max_date = datetime(2024, 12, 31)
    
    start_date = st.sidebar.date_input(
        "开始日期",
        value=datetime(2015, 1, 5),
        min_value=min_date,
        max_value=max_date
    )
    
    end_date = st.sidebar.date_input(
        "结束日期",
        value=datetime(2023, 12, 29),
        min_value=start_date,
        max_value=max_date
    )
    
    # 策略参数
    hold_count = st.sidebar.number_input(
        "持仓数量",
        min_value=1,
        max_value=500,
        value=50
    )
    
    rebalance_frequency = st.sidebar.number_input(
        "再平衡频率(天)",
        min_value=1,
        max_value=30,
        value=1
    )
    
    # 策略选择
    run_fixed = st.sidebar.checkbox("运行固定持仓策略", value=True)
    run_dynamic = st.sidebar.checkbox("运行动态持仓策略", value=True)
    
    # 运行回测按钮
    if st.sidebar.button("开始回测"):
        with st.spinner("正在进行回测..."):
            try:
                params = {
                    'start_date': start_date.strftime("%Y-%m-%d"),
                    'end_date': end_date.strftime("%Y-%m-%d"),
                    'hold_count': hold_count,
                    'rebalance_frequency': rebalance_frequency,
                    'run_fixed': run_fixed,
                    'run_dynamic': run_dynamic
                }
                
                fixed_results, dynamic_results = main(**params)
                
                # 显示固定持仓策略结果
                if run_fixed and fixed_results is not None:
                    st.header("固定持仓策略结果")
                    display_metrics(fixed_results, "固定持仓")
                    fig_fixed = plot_cumulative_returns(fixed_results, "固定持仓")
                    if fig_fixed:
                        st.plotly_chart(fig_fixed, use_container_width=True)
                
                # 显示动态持仓策略结果
                if run_dynamic and dynamic_results is not None:
                    st.header("动态持仓策略结果")
                    display_metrics(dynamic_results, "动态持仓")
                    fig_dynamic = plot_cumulative_returns(dynamic_results, "动态持仓")
                    if fig_dynamic:
                        st.plotly_chart(fig_dynamic, use_container_width=True)
                
            except Exception as e:
                st.error(f"回测过程中出现错误: {str(e)}")

if __name__ == "__main__":
    main_app() 