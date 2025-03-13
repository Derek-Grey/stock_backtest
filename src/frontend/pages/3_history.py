# -*- coding: utf-8 -*-
# title: 回测记录
"""
回测记录页面
"""
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from pathlib import Path
import sys
import pymongo

# 添加项目根目录到系统路径
ROOT_DIR = Path(__file__).resolve().parent.parent.parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

def load_results():
    """加载历史回测结果"""
    results_dir = Path(ROOT_DIR) / "results"
    if not results_dir.exists():
        return []
    
    results = []
    for file in results_dir.glob("*_strategy_*.csv"):
        strategy_type = "固定持仓" if "fixed" in file.name else "动态持仓"
        timestamp = file.name.split("_")[-1].replace(".csv", "")
        results.append({
            "文件名": file.name,
            "策略类型": strategy_type,
            "回测时间": timestamp,
            "路径": file
        })
    
    return results

def plot_result(df):
    """绘制回测结果图表"""
    if 'daily_return' not in df.columns:
        st.error("数据格式不正确")
        return
    
    cumulative_returns = (1 + df['daily_return']).cumprod() - 1
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=cumulative_returns.index,
        y=cumulative_returns.values,
        mode='lines',
        name='累计收益率'
    ))
    
    fig.update_layout(
        title='策略累计收益率',
        xaxis_title='日期',
        yaxis_title='累计收益率',
        yaxis_tickformat='.2%'
    )
    
    return fig

def history_page():
    st.title("📝 回测记录")
    
    st.markdown(
        """
        <style>
        .main-title {
            font-size: 2rem;  /* Unified font size */
        }
        </style>
        """, unsafe_allow_html=True
    )
    
    results = load_results()
    
    if not results:
        st.info("暂无回测记录")
        return
    
    # 创建数据筛选器
    strategy_types = list(set(r["策略类型"] for r in results))
    selected_strategy = st.selectbox("选择策略类型", ["全部"] + strategy_types)
    
    # 筛选结果
    filtered_results = [
        r for r in results 
        if selected_strategy == "全部" or r["策略类型"] == selected_strategy
    ]
    
    # 显示结果列表
    for result in filtered_results:
        with st.expander(f"{result['策略类型']} - {result['回测时间']}"):
            try:
                df = pd.read_csv(result['路径'], index_col=0)
                df.index = pd.to_datetime(df.index)
                
                # 显示回测结果图表
                fig = plot_result(df)
                st.plotly_chart(fig, use_container_width=True)
                
                # 显示回测数据
                if st.checkbox("查看详细数据", key=result['文件名']):
                    st.dataframe(df)
                
                # 下载按钮
                st.download_button(
                    label="下载数据",
                    data=df.to_csv(),
                    file_name=result['文件名'],
                    mime="text/csv"
                )
                
            except Exception as e:
                st.error(f"加载结果文件时出错: {str(e)}")

if __name__ == "__main__":
    history_page() 