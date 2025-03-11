# -*- coding: utf-8 -*-
# title: 策略回测
"""
策略回测页面
集成现有回测系统的前端界面
"""
import sys
from pathlib import Path
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime
from loguru import logger
import pymongo

# 添加项目根目录到系统路径
ROOT_DIR = Path(__file__).resolve().parent.parent.parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

from src.backtest.main import main, save_results
from config.settings import DATA_DIR, OUTPUT_DIR

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
    
    col1, col2, col3 = st.columns(3)
    
    # 计算关键指标
    total_return = (1 + daily_returns).cumprod().iloc[-1] - 1
    annual_return = (1 + total_return) ** (252 / len(daily_returns)) - 1
    volatility = daily_returns.std() * (252 ** 0.5)
    sharpe = annual_return / volatility if volatility != 0 else 0
    max_drawdown = ((1 + daily_returns).cumprod() / 
                    (1 + daily_returns).cumprod().cummax() - 1).min()
    
    with col1:
        st.metric("总收益率", f"{total_return:.2%}")
        st.metric("年化收益率", f"{annual_return:.2%}")
    
    with col2:
        st.metric("年化波动率", f"{volatility:.2%}")
        st.metric("夏普比率", f"{sharpe:.2f}")
    
    with col3:
        st.metric("最大回撤", f"{max_drawdown:.2%}")

def render_backtest_page():
    """渲染回测页面"""
    st.title("策略回测系统")
    
    with st.sidebar:
        st.header("回测参数设置")
        
        # 日期选择
        col1, col2 = st.columns(2)
        with col1:
            start_date = st.date_input(
                "开始日期",
                value=pd.to_datetime("2015-01-05"),
                min_value=pd.to_datetime("2010-01-01"),
                max_value=pd.to_datetime("2023-12-31")
            )
        with col2:
            end_date = st.date_input(
                "结束日期",
                value=pd.to_datetime("2023-12-29"),
                min_value=pd.to_datetime("2010-01-01"),
                max_value=pd.to_datetime("2023-12-31")
            )
        
        # 持仓方式选择
        position_type = st.radio(
            "持仓方式",
            ["固定数量", "动态百分比"],
            index=0
        )
        
        if position_type == "固定数量":
            # 固定持仓参数
            hold_count = st.number_input("持仓数量", min_value=1, max_value=500, value=50)
            rebalance_frequency = st.number_input("再平衡频率(天)", min_value=1, max_value=30, value=1)
        else:
            # 动态持仓参数
            start_percentage = st.number_input("起始百分比", min_value=0.0, max_value=1.0, value=0.01, format="%.3f")
            end_percentage = st.number_input("结束百分比", min_value=0.0, max_value=1.0, value=0.03, format="%.3f")
            rebalance_frequency = st.number_input("再平衡频率(天)", min_value=1, max_value=30, value=1)
        
        # 策略选择
        run_fixed = st.checkbox("运行固定持仓策略", value=True)
        run_dynamic = st.checkbox("运行动态持仓策略", value=True)
        
        # 运行按钮
        run_button = st.button("开始回测")

    # 主界面
    if run_button:
        if not (run_fixed or run_dynamic):
            st.error("请至少选择一个回测策略！")
            return
            
        if start_date >= end_date:
            st.error("开始日期必须早于结束日期！")
            return
            
        try:
            with st.spinner("正在执行回测..."):
                params = {
                    'start_date': start_date.strftime("%Y-%m-%d"),
                    'end_date': end_date.strftime("%Y-%m-%d"),
                    'rebalance_frequency': rebalance_frequency,
                    'run_fixed': run_fixed,
                    'run_dynamic': run_dynamic,
                    'position_type': position_type
                }
                
                if position_type == "固定数量":
                    params['hold_count'] = hold_count
                else:
                    params['start_percentage'] = start_percentage 
                    params['end_percentage'] = end_percentage
                
                fixed_results, dynamic_results = main(**params)
                
                # 显示回测结果
                if fixed_results is not None and run_fixed:
                    st.subheader("固定持仓策略结果")
                    display_metrics(fixed_results, "固定持仓")
                    fig = plot_cumulative_returns(fixed_results, "固定持仓")
                    if fig:
                        st.plotly_chart(fig, use_container_width=True)
                
                if dynamic_results is not None and run_dynamic:
                    st.subheader("动态持仓策略结果")
                    display_metrics(dynamic_results, "动态持仓")
                    fig = plot_cumulative_returns(dynamic_results, "动态持仓")
                    if fig:
                        st.plotly_chart(fig, use_container_width=True)
                
                # 保存回测结果
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                if fixed_results is not None:
                    save_results(fixed_results, f"fixed_{timestamp}", OUTPUT_DIR)
                if dynamic_results is not None:
                    save_results(dynamic_results, f"dynamic_{timestamp}", OUTPUT_DIR)
                
                # 显示详细数据
                st.subheader("详细回测数据")
                if fixed_results is not None and run_fixed:
                    with st.expander("固定持仓策略详细数据"):
                        st.dataframe(fixed_results)
                        
                if dynamic_results is not None and run_dynamic:
                    with st.expander("动态持仓策略详细数据"):
                        st.dataframe(dynamic_results)
                
        except Exception as e:
            st.error(f"回测执行失败: {str(e)}")
            logger.exception("回测失败详细信息:")

if __name__ == "__main__":
    # 配置日志
    logger.add(
        Path(ROOT_DIR) / "logs/backtest_frontend_{time}.log",
        rotation="1 day",
        retention="7 days",
        level="INFO"
    )
    render_backtest_page() 