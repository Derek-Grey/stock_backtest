# -*- coding: utf-8 -*-
# title: 数据查看
"""
数据查看页面
"""
import streamlit as st
import pandas as pd
from pathlib import Path
import sys
import pymongo

# 添加项目根目录到系统路径
ROOT_DIR = Path(__file__).resolve().parent.parent.parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

from src.data.load_data import LoadData

def data_view_page():
    st.title("📊 数据查看")
    
    # Custom CSS for styling
    st.markdown(
        """
        <style>
        .stButton>button {
            background-color: #4CAF50;
            color: white;
            border-radius: 8px;
            padding: 10px 24px;
            font-size: 16px;
        }
        .stDateInput>div {
            border-radius: 8px;
        }
        </style>
        """, unsafe_allow_html=True
    )
    
    # 使用列布局
    col1, col2 = st.columns(2)
    
    with col1:
        # 日期选择
        date_s = st.date_input("选择开始日期", value=pd.to_datetime("2010-01-01"))
    
    with col2:
        date_e = st.date_input("选择结束日期", value=pd.to_datetime("2024-12-31"))
    
    # 数据类型选择
    st.markdown("### 选择数据类型")
    data_type = st.selectbox(
        "",
        ["股票数据", "交易状态", "风险警示", "涨跌停", "评分矩阵"]
    )
    
    # 确认按钮
    if st.button("🔍 确认查询"):
        try:
            with st.spinner('加载数据中，请稍候...'):
                # 创建数据加载器实例
                data_loader = LoadData(
                    date_s=str(date_s),
                    date_e=str(date_e),
                    data_folder=str(ROOT_DIR / "data")
                )
                
                if data_type == "股票数据":
                    df_stocks, _, _, _ = data_loader.get_stocks_info()
                    st.dataframe(df_stocks)
                    
                elif data_type == "交易状态":
                    _, trade_status, _, _ = data_loader.get_stocks_info()
                    st.dataframe(trade_status)
                    
                elif data_type == "风险警示":
                    _, _, risk_warning, _ = data_loader.get_stocks_info()
                    st.dataframe(risk_warning)
                    
                elif data_type == "涨跌停":
                    _, _, _, limit = data_loader.get_stocks_info()
                    st.dataframe(limit)
                    
                elif data_type == "评分矩阵":
                    score_matrix = data_loader.generate_score_matrix('stra_V3_11.csv')
                    st.dataframe(score_matrix)
                
        except Exception as e:
            st.error(f"加载数据时出错: {str(e)}")

if __name__ == "__main__":
    data_view_page() 