"""
回测模块
包含回测类和绘图类
"""
import os
import time
import pandas as pd
import numpy as np
from loguru import logger
import plotly.graph_objects as go
from pathlib import Path
from functools import wraps

from src.utils.decorators import cache_data, log_function_call
from src.utils.helpers import trans_str_to_float64
from config.settings import OUTPUT_DIR

class Backtest:
    """
    股票回测策略类，支持固定持仓和动态持仓两种策略
    
    Attributes:
        stocks_matrix: 股票收益率矩阵
        limit_matrix: 涨跌停限制矩阵
        risk_warning_matrix: 风险警示矩阵
        trade_status_matrix: 交易状态矩阵
        score_matrix: 股票评分矩阵
        output_dir: 输出目录
    """
    
    def __init__(self, stocks_matrix, limit_matrix, risk_warning_matrix, 
                 trade_status_matrix, score_matrix, output_dir=OUTPUT_DIR):
        """初始化回测类"""
        self.stocks_matrix = stocks_matrix
        self.limit_matrix = limit_matrix
        self.risk_warning_matrix = risk_warning_matrix
        self.trade_status_matrix = trade_status_matrix
        self.score_matrix = score_matrix
        self.output_dir = Path(output_dir)
        
        # 创建输出目录
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # 生成有效性矩阵
        self._generate_validity_matrices()
        self.plotter = StrategyPlotter(output_dir)
    
    def _generate_validity_matrices(self):
        """生成有效性矩阵和受限股票矩阵"""
        # 生成有效性矩阵：股票必须同时满足三个条件
        self.risk_warning_validity = (self.risk_warning_matrix == 0).astype(int)
        self.trade_status_validity = (self.trade_status_matrix == 1).astype(int)
        self.limit_validity = (self.limit_matrix == 0).astype(int)
        
        self.valid_stocks_matrix = (
            self.risk_warning_validity * 
            self.trade_status_validity * 
            self.limit_validity
        )
        
        # 受限股票矩阵：只考虑交易状态和涨跌停限制
        self.restricted_stocks_matrix = (
            self.trade_status_validity * self.limit_validity
        )
    
    @log_function_call
    def _update_positions(self, position_history, day, hold_count, 
                         rebalance_frequency, strategy_type, current_start_pos=None):
        """更新持仓策略的持仓"""
        previous_positions = position_history.iloc[day - 1]["hold_positions"]
        current_date = position_history.index[day]
        current_hold_count = hold_count

        # 动态策略处理
        if strategy_type == "dynamic":
            current_hold_count = max(1, int(current_hold_count))

        # 评分矩阵空值检查
        if self.score_matrix.iloc[day - 1].isna().all():
            position_history.loc[current_date, "hold_positions"] = previous_positions
            return

        # 处理前一天的持仓
        previous_positions = (set() if pd.isna(previous_positions) 
                            else set(previous_positions.split(',')))
        previous_positions = {stock for stock in previous_positions 
                            if isinstance(stock, str) and stock.isalnum()}

        # 计算有效股票和受限股票
        valid_stocks = self.valid_stocks_matrix.iloc[day].astype(bool)
        restricted = self.restricted_stocks_matrix.iloc[day].astype(bool)
        previous_date = position_history.index[day - 1]
        valid_scores = self.score_matrix.loc[previous_date]

        # 处理受限股票
        restricted_stocks = [stock for stock in previous_positions 
                           if not restricted[stock]]

        # 再平衡处理
        if (day - 1) % rebalance_frequency == 0:
            sorted_stocks = valid_scores.sort_values(ascending=False)
            try:
                if strategy_type == "fixed":
                    top_stocks = sorted_stocks.iloc[:hold_count]
                else:  # dynamic
                    start_pos = max(0, int(current_start_pos))
                    hold_num = max(1, int(current_hold_count))
                    top_stocks = sorted_stocks.iloc[start_pos:start_pos + hold_num]

                retained_stocks = list(set(previous_positions) & 
                                    set(top_stocks) | 
                                    set(restricted_stocks))
                new_positions_needed = hold_count - len(retained_stocks)
                final_positions = set(retained_stocks)

                if new_positions_needed > 0:
                    new_stocks = sorted_stocks[valid_stocks].index
                    new_stocks = [stock for stock in new_stocks 
                                if stock not in final_positions]
                    final_positions.update(new_stocks[:new_positions_needed])
            except IndexError:
                logger.warning(f"日期 {current_date}: 可用股票数量不足，使用所有有效股票")
                final_positions = set(sorted_stocks[valid_stocks].index[:hold_count])
        else:
            final_positions = set(previous_positions)

        # 更新持仓信息
        position_history.loc[current_date, "hold_positions"] = ','.join(final_positions)

        # 计算每日收益率和换手率
        if previous_date in self.stocks_matrix.index:
            daily_returns = self.stocks_matrix.loc[current_date, 
                                                 list(final_positions)].astype(float)
            position_history.loc[current_date, "daily_return"] = daily_returns.mean()

        turnover_rate = (len(previous_positions - final_positions) / 
                        max(len(previous_positions), 1))
        position_history.at[current_date, "turnover_rate"] = turnover_rate

    @log_function_call
    def run_fixed_strategy(self, hold_count, rebalance_frequency, 
                          strategy_name="fixed"):
        """运行固定持仓策略"""
        start_time = time.time()
        position_history = self._initialize_position_history(strategy_name)

        for day in range(1, len(self.stocks_matrix)):
            self._update_positions(position_history, day, hold_count, 
                                 rebalance_frequency, "fixed")
        
        return self._process_results(position_history, strategy_name, start_time)

    @log_function_call
    def run_dynamic_strategy(self, rebalance_frequency, df_mv, 
                           start_sorted=100, the_end_month=None, 
                           fixed_by_month=True):
        """运行动态持仓策略"""
        start_time = time.time()
        position_history = self._initialize_position_history("dynamic")

        for day in range(1, len(self.stocks_matrix)):
            current_date = position_history.index[day].strftime('%Y-%m-%d')
            current_hold_count = (df_mv.loc[current_date, 'hold_num'] 
                                if current_date in df_mv.index else 50)
            
            self._update_positions(position_history, day, int(current_hold_count),
                                 rebalance_frequency, "dynamic", start_sorted)
        
        return self._process_results(position_history, "dynamic", start_time)

    def _initialize_position_history(self, strategy_name):
        """初始化持仓历史DataFrame"""
        return pd.DataFrame(
            index=self.stocks_matrix.index,
            columns=["hold_positions", "daily_return", "strategy"]
        ).assign(strategy=strategy_name)

    def _process_results(self, position_history, strategy_name, start_time):
        """处理回测结果"""
        # 删除没有持仓记录的行
        position_history = position_history.dropna(subset=["hold_positions"])

        # 计算持仓数量
        position_history = position_history.copy()  # 创建副本
        position_history.loc[:, 'hold_count'] = position_history['hold_positions'].apply(
            lambda x: len(x.split(',')) if pd.notna(x) else 0
        )
        
        # 保存结果
        results = position_history[['hold_positions', 'hold_count', 
                                  'turnover_rate', 'daily_return']]
        results.index.name = 'date'
        
        csv_file = self.output_dir / f'strategy_results_{strategy_name}.csv'
        results.to_csv(csv_file)
        
        # 计算统计指标
        self._log_statistics(results, strategy_name, start_time)
        
        return results

    def _log_statistics(self, results, strategy_name, start_time):
        """输出统计信息"""
        stats = {
            'cumulative_return': (1 + results['daily_return']).cumprod().iloc[-1] - 1,
            'avg_daily_return': results['daily_return'].mean(),
            'avg_turnover': results['turnover_rate'].mean(),
            'avg_holdings': results['hold_count'].mean()
        }
        
        logger.info(f"\n=== {strategy_name}策略统计 ===")
        logger.info(f"累计收益率: {stats['cumulative_return']:.2%}")
        logger.info(f"平均日收益率: {stats['avg_daily_return']:.2%}")
        logger.info(f"平均换手率: {stats['avg_turnover']:.2%}")
        logger.info(f"平均持仓量: {stats['avg_holdings']:.1f}")
        logger.info(f"策略运行耗时: {time.time() - start_time:.2f} 秒")

    def plot_results(self, results, strategy_type, turn_loss=0.003):
        """绘制策略结果图表"""
        self.plotter.plot_net_value(results, strategy_type, turn_loss)


class StrategyPlotter:
    """策略结果可视化类"""
    
    def __init__(self, output_dir):
        """初始化绘图类"""
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def plot_net_value(self, df: pd.DataFrame, strategy_name: str, 
                      turn_loss: float = 0.003):
        """绘制策略的累计净值和回撤曲线"""
        df = df.copy()
        df.reset_index(inplace=True)
        df.set_index('date', inplace=True)
        
        # 计算成本和净值
        self._calculate_costs_and_returns(df, turn_loss)
        
        # 计算回撤
        self._calculate_drawdown(df)
        
        # 计算统计指标
        stats = self._calculate_statistics(df)
        
        # 创建图表
        self._create_plot(df, strategy_name, df.index[0], stats)
    
    def _calculate_costs_and_returns(self, df: pd.DataFrame, turn_loss: float):
        """计算成本和收益"""
        df['loss'] = 0.0013
        df.loc[df.index > '2023-08-31', 'loss'] = 0.0008
        df['loss'] += float(turn_loss)
        
        df['chg_'] = df['daily_return'] - df['turnover_rate'] * df['loss']
        df['net_value'] = (df['chg_'] + 1).cumprod()
    
    def _calculate_drawdown(self, df: pd.DataFrame):
        """计算最大回撤"""
        df['max_net'] = df['net_value'].expanding().max()
        df['back_net'] = df['net_value'] / df['max_net'] - 1
    
    def _calculate_statistics(self, df: pd.DataFrame):
        """计算统计指标"""
        # 确保 net_value 列是浮点数类型
        net_value = df['net_value'].astype(float)
        # 计算百分比变化
        volatility = net_value.pct_change()
        
        return {
            'annualized_return': format(net_value.iloc[-1] ** 
                                      (252 / len(df)) - 1, '.2%'),
            'monthly_volatility': format(volatility.std() * 
                                       21 ** 0.5, '.2%'),
            'end_date': df.index[-1]
        }
    
    def _create_plot(self, df: pd.DataFrame, strategy_name: str, start_date, stats: dict):
        """创建图表"""
        # 设置更鲜艳的颜色
        fig = go.Figure(data=[
            go.Scatter(
                x=df.index, 
                y=df['net_value'], 
                name='净值',
                line=dict(color='#FF4B4B', width=2)  # 鲜艳的红色，加粗线条
            ),
            go.Scatter(
                x=df.index, 
                y=df['back_net'] * 100, 
                name='回撤',
                xaxis='x', 
                yaxis='y2', 
                mode="none", 
                fill="tozeroy",
                fillcolor='rgba(65, 105, 225, 0.3)'  # 半透明的皇家蓝
            )
        ])

        fig.update_layout(
            height=1122,
            title=f"{strategy_name}策略，<br>"
                  f"净值（左）& 回撤（右），<br>"
                  f"全期：{start_date} ~ {stats['end_date']}，<br>"
                  f"年化收益：{stats['annualized_return']}，"
                  f"月波动：{stats['monthly_volatility']}",
            font={"size": 22},
            yaxis=dict(
                title=dict(
                    text="累计净值",
                    font=dict(color="#FF4B4B")  # 与净值线条颜色匹配
                ),
                tickfont=dict(color="#FF4B4B")
            ),
            yaxis2=dict(
                title=dict(
                    text="最大回撤",
                    font=dict(color="royalblue")  # 与回撤颜色匹配
                ),
                side="right",
                overlaying="y",
                ticksuffix="%",
                showgrid=False,
                tickfont=dict(color="royalblue")
            ),
            plot_bgcolor='white',  # 白色背景
            paper_bgcolor='white',
            showlegend=True,
            legend=dict(
                yanchor="top",
                y=0.99,
                xanchor="left",
                x=0.01,
                bgcolor='rgba(255, 255, 255, 0.8)'  # 半透明的白色背景
            ),
            margin=dict(t=150)  # 增加顶部边距，让标题显示更完整
        )

        # 添加网格线
        fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='lightgray')
        fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='lightgray')

        fig.show() 