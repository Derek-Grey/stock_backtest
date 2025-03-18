# -*- coding: utf-8 -*-
# title: 权重调整
"""
权重调整页面
允许用户上传权重CSV文件，设置每日变化限制，并生成符合限制的新权重文件
"""
import sys
from pathlib import Path
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime
from loguru import logger

# 添加项目根目录到系统路径
ROOT_DIR = Path(__file__).resolve().parent.parent.parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

from config.settings import DATA_DIR, OUTPUT_DIR

class PortfolioWeightAdjuster:
    def __init__(self, df, change_limit=0.05):
        self.df = df.copy()
        self.change_limit = change_limit
        
        # 检查是否有 'datetime' 列，并将其转换为日期时间格式
        if 'datetime' in self.df.columns:
            self.df['datetime'] = pd.to_datetime(self.df['datetime'])
            self.time_column = 'datetime'
        else:
            self.df['date'] = pd.to_datetime(self.df['date'])
            self.time_column = 'date'
        
        # 获取所有代码的集合并排序
        self.all_codes = sorted(set(self.df['code']))
        # 获取时间列的唯一值
        self.time_values = sorted(self.df[self.time_column].unique())

    def validate_weights_sum(self) -> bool:
        """验证CSV文件中每个时间点的权重和是否为1"""
        try:
            grouped = self.df.groupby(self.time_column)['weight'].sum()
            invalid_times = []
            for time_value, weight_sum in grouped.items():
                if not (0.999 <= weight_sum <= 1.001):
                    invalid_times.append((time_value, weight_sum))
            
            if invalid_times:
                return False, invalid_times
            return True, []
        except Exception as e:
            return False, str(e)

    def get_target_weights_from_csv(self):
        """从CSV文件中提取目标权重"""
        target_weights_list = []
        codes_list = []
        for time_value in self.time_values:
            group = self.df[self.df[self.time_column] == time_value]
            # 创建一个包含所有代码的字典，默认权重为0
            weights_dict = {code: 0 for code in self.all_codes}
            # 更新有权重的代码
            for _, row in group.iterrows():
                weights_dict[row['code']] = row['weight']
            
            target_weights = [weights_dict[code] for code in self.all_codes]
            codes = self.all_codes.copy()
            
            target_weights_list.append(target_weights)
            codes_list.append(codes)
        return target_weights_list, codes_list

    def get_initial_weights(self):
        """从CSV文件中提取初始权重"""
        first_time_value = self.time_values[0]
        group = self.df[self.df[self.time_column] == first_time_value]
        
        # 创建一个包含所有代码的字典，默认权重为0
        weights_dict = {code: 0 for code in self.all_codes}
        # 更新有权重的代码
        for _, row in group.iterrows():
            weights_dict[row['code']] = row['weight']
        
        initial_weights = [weights_dict[code] for code in self.all_codes]
        return initial_weights

    def adjust_weights_over_days(self, current_weights, target_weights_list, codes_list):
        """调整当前权重向多个目标权重靠近，具有变化限制。"""
        adjusted_weights_list = []
        for target_weights, codes in zip(target_weights_list, codes_list):
            adjusted_weights = []
            for code in self.all_codes:
                if code in codes:
                    target_index = codes.index(code)
                    target_weight = target_weights[target_index]
                else:
                    target_weight = 0

                current_index = self.all_codes.index(code)
                current_weight = current_weights[current_index] if current_index < len(current_weights) else 0

                weight_change = target_weight - current_weight
                if abs(weight_change) > self.change_limit:
                    weight_change = self.change_limit if weight_change > 0 else -self.change_limit

                adjusted_weight = current_weight + weight_change
                # 确保权重不为负
                adjusted_weight = max(0, adjusted_weight)
                adjusted_weights.append(adjusted_weight)

            # 归一化调整后的权重，使其总和为1
            total_weight = sum(adjusted_weights)
            if total_weight > 0:  # 避免除以零
                adjusted_weights = [w / total_weight for w in adjusted_weights]
            
            current_weights = adjusted_weights
            adjusted_weights_list.append(adjusted_weights)

        return adjusted_weights_list

    def create_adjusted_weights_df(self, adjusted_weights_list):
        """创建包含调整后权重的DataFrame"""
        rows = []
        for time_idx, time_value in enumerate(self.time_values):
            weights = adjusted_weights_list[time_idx]
            for code_idx, code in enumerate(self.all_codes):
                weight = weights[code_idx]
                if weight > 0:  # 只包含非零权重
                    rows.append({
                        self.time_column: time_value,
                        'code': code,
                        'weight': weight
                    })
        return pd.DataFrame(rows)

    def plot_adjusted_weight_sums(self, adjusted_weights_list):
        """使用plotly绘制调整后的权重和随时间的变化图"""
        try:
            # 计算每个时间点调整后的权重和
            adjusted_sums = [sum(weights) for weights in adjusted_weights_list]
            
            # 创建图形
            fig = go.Figure()
            
            # 添加实际权重和的线
            fig.add_trace(
                go.Scatter(
                    x=self.time_values,
                    y=adjusted_sums,
                    mode='lines+markers',
                    name='实际权重和',
                    line=dict(color='#2E86C1', width=2),
                    marker=dict(
                        size=8,
                        color='white',
                        line=dict(color='#2E86C1', width=2)
                    )
                )
            )
            
            # 添加目标权重和的参考线
            fig.add_hline(
                y=1.0,
                line=dict(color='#E74C3C', dash='dash'),
                opacity=0.5,
                name='目标权重和'
            )
            
            # 更新布局
            fig.update_layout(
                title={
                    'text': '调整后权重和变化',
                    'y': 0.95,
                    'x': 0.5,
                    'xanchor': 'center',
                    'yanchor': 'top',
                    'font': dict(size=20)
                },
                xaxis_title='时间',
                yaxis_title='权重和',
                template='plotly_white',
                showlegend=True,
                legend=dict(
                    yanchor="top",
                    y=0.99,
                    xanchor="left",
                    x=0.01
                ),
                xaxis=dict(
                    tickangle=30,
                    tickformat='%Y-%m-%d'
                ),
                hovermode='x unified'
            )
            
            return fig
            
        except Exception as e:
            logger.exception(f"绘制调整后权重和图时出错：{e}")
            return None

def weight_adjuster_page():
    st.title("⚖️ 权重调整")
    
    st.markdown(
        """
        <style>
        .main-title {
            font-size: 2rem;
        }
        </style>
        """, unsafe_allow_html=True
    )
    
    st.markdown("""
    ### 功能说明
    
    此工具用于调整投资组合权重，确保每日权重变化不超过设定的限制。
    
    1. 上传包含目标权重的CSV文件
    2. 设置每日最大权重变化限制
    3. 系统将生成符合限制的新权重文件
    """)
    
    # CSV Template Download
    st.markdown("### 下载CSV模板")
    st.download_button(
        label="下载权重CSV模板",
        data="date,code,weight\n2023-01-03,SH600788,0.1\n2023-01-03,SZ000765,0.2\n2023-01-04,SH600788,0.15\n2023-01-04,SZ000765,0.25\n",
        file_name="weight_template.csv",
        mime="text/csv"
    )
    
    # 文件上传
    weight_file = st.file_uploader("上传权重矩阵CSV文件", type=['csv'])
    
    # 设置每日变化限制
    change_limit = st.slider(
        "设置每日最大权重变化限制", 
        min_value=0.01, 
        max_value=0.20, 
        value=0.05, 
        step=0.01,
        format="%.2f"
    )
    
    if weight_file is not None:
        try:
            # 读取CSV文件
            weight_df = pd.read_csv(weight_file)
            
            # 验证文件格式
            required_columns = ['date', 'code', 'weight']
            if not all(col in weight_df.columns for col in required_columns):
                missing_cols = [col for col in required_columns if col not in weight_df.columns]
                st.error(f"权重CSV文件缺少必需列: {', '.join(missing_cols)}")
                return
            
            # 检查日期格式
            try:
                weight_df['date'] = pd.to_datetime(weight_df['date'])
            except:
                st.error("日期列格式不正确，应为YYYY-MM-DD格式")
                return
            
            # 创建权重调整器
            adjuster = PortfolioWeightAdjuster(weight_df, change_limit)
            
            # 验证权重和
            is_valid, invalid_data = adjuster.validate_weights_sum()
            if not is_valid:
                if isinstance(invalid_data, str):
                    st.error(f"权重验证失败: {invalid_data}")
                else:
                    st.error("以下日期的权重和不为1:")
                    for time_value, weight_sum in invalid_data:
                        st.error(f"  - {time_value}: {weight_sum:.4f}")
                    
                    # 提供修复选项
                    if st.button("自动修复权重和"):
                        # 对每个时间点的权重进行归一化
                        normalized_rows = []
                        for time_value, group in weight_df.groupby('date'):
                            total_weight = group['weight'].sum()
                            if total_weight > 0:  # 避免除以零
                                for _, row in group.iterrows():
                                    normalized_row = row.copy()
                                    normalized_row['weight'] = row['weight'] / total_weight
                                    normalized_rows.append(normalized_row)
                        
                        if normalized_rows:
                            weight_df = pd.DataFrame(normalized_rows)
                            adjuster = PortfolioWeightAdjuster(weight_df, change_limit)
                            st.success("权重已自动归一化")
                        else:
                            st.error("无法修复权重，请检查数据")
                            return
                    else:
                        return
            
            # 添加确认按钮
            if st.button("开始调整权重"):
                with st.spinner("正在调整权重..."):
                    # 获取目标权重和初始权重
                    target_weights_list, codes_list = adjuster.get_target_weights_from_csv()
                    initial_weights = adjuster.get_initial_weights()
                    
                    # 调整权重
                    adjusted_weights_list = adjuster.adjust_weights_over_days(
                        initial_weights, target_weights_list, codes_list
                    )
                    
                    # 创建调整后的权重DataFrame
                    adjusted_df = adjuster.create_adjusted_weights_df(adjusted_weights_list)
                    
                    # 显示调整结果
                    st.subheader("调整结果")
                    
                    # 显示权重和图表
                    fig = adjuster.plot_adjusted_weight_sums(adjusted_weights_list)
                    if fig:
                        st.plotly_chart(fig, use_container_width=True)
                    
                    # 显示调整后的权重表格
                    st.dataframe(adjusted_df)
                    
                    # 提供下载选项
                    csv = adjusted_df.to_csv(index=False)
                    st.download_button(
                        label="下载调整后的权重CSV",
                        data=csv,
                        file_name="adjusted_weights.csv",
                        mime="text/csv"
                    )
                    
                    # 保存到本地文件
                    output_path = Path(OUTPUT_DIR) / f"adjusted_weights_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
                    adjusted_df.to_csv(output_path, index=False)
                    st.success(f"调整后的权重已保存到: {output_path}")
                    
        except Exception as e:
            st.error(f"文件处理失败: {str(e)}")
            logger.exception("权重调整失败详细信息:")

if __name__ == "__main__":
    weight_adjuster_page() 