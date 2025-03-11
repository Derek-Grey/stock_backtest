"""
数据加载模块
用于从数据库加载和处理回测所需的数据
"""
import os
import math
import pandas as pd
import numpy as np
from loguru import logger
from pathlib import Path
from functools import wraps

from src.data.db_client import get_client_U, get_client
from src.utils.decorators import cache_data
from src.utils.helpers import trans_str_to_float64
from config.settings import USELESS_INDUS, END_MONTH

class LoadData:
    """数据加载类"""
    
    def __init__(self, date_s: str, date_e: str, data_folder: str):
        """
        初始化数据加载器
        
        Args:
            date_s: 开始日期
            date_e: 结束日期
            data_folder: 数据存储目录
        """
        if date_s is None or date_e is None:
            raise ValueError('必须指定起止日期！！！')
            
        self.client_U = get_client_U()
        self.date_s = date_s
        self.date_e = date_e
        self.data_folder = data_folder
        os.makedirs(self.data_folder, exist_ok=True)

    def _validate_date_range(self, df):
        """验证数据日期范围是否满足要求"""
        dates = pd.to_datetime(df.index)
        start = pd.to_datetime(self.date_s)
        end = pd.to_datetime(self.date_e)
        return dates.min() <= start and dates.max() >= end

    def get_chg_wind(self) -> pd.DataFrame:
        """获取WIND的日频涨跌幅矩阵"""
        matrix_path = Path(self.data_folder) / 'chg_matrix.pkl'
        
        # 如果本地文件存在,尝试加载
        if matrix_path.exists():
            try:
                df = pd.read_pickle(matrix_path)
                if self._validate_date_range(df):
                    logger.debug(f"从本地文件 {matrix_path} 加载涨跌幅矩阵")
                    return df
            except Exception as e:
                logger.error(f'读取矩阵文件失败: {e}')

        # 从数据库加载并处理数据
        logger.debug('从数据库加载WIND日频涨跌幅数据...')
        df = pd.DataFrame(
            self.client_U.basic_wind.w_vol_price.find(
                {"date": {"$gte": self.date_s, "$lte": self.date_e}},
                {"_id": 0, "date": 1, "code": 1, "pct_chg": 1},
                batch_size=1000000
            )
        )
        df = trans_str_to_float64(df, trans_cols=['pct_chg'])
        df['date'] = pd.to_datetime(df['date'])
        
        # 转换为矩阵并保存
        matrix = df.pivot_table(index='date', columns='code', values='pct_chg')
        matrix.to_pickle(matrix_path)
        return matrix

    def get_stocks_info(self) -> tuple:
        """获取股票信息矩阵"""
        matrix_paths = {
            'trade_status': Path(self.data_folder) / 'trade_status_matrix.pkl',
            'riskwarning': Path(self.data_folder) / 'riskwarning_matrix.pkl',
            'limit': Path(self.data_folder) / 'limit_matrix.pkl'
        }
        
        # 检查所有矩阵文件是否存在且有效
        matrices = {}
        all_valid = True
        for name, path in matrix_paths.items():
            if path.exists():
                try:
                    matrix = pd.read_pickle(path)
                    if self._validate_date_range(matrix):
                        matrices[name] = matrix
                    else:
                        all_valid = False
                        break
                except Exception as e:
                    logger.error(f'读取{name}矩阵文件失败: {e}')
                    all_valid = False
                    break
            else:
                all_valid = False
                break

        if all_valid:
            logger.debug("从本地文件加载所有矩阵数据")
            df_stocks = self.get_chg_wind()
            return (df_stocks, matrices['trade_status'], 
                   matrices['riskwarning'], matrices['limit'])

        # 从数据库加载数据
        logger.debug('从数据库加载股票信息...')
        # 获取基本信息
        df_info = pd.DataFrame(
            self.client_U.basic_wind.w_basic_info.find(
                {"date": {"$gte": self.date_s, "$lte": self.date_e}},
                {"_id": 0, 'date': 1, 'code': 1, 'riskwarning': 1, 'trade_status': 1},
                batch_size=1000000
            )
        )
        df_info['date'] = pd.to_datetime(df_info['date'])
        
        # 获取涨跌停信息
        df_limit = pd.DataFrame(
            self.client_U.basic_jq.jq_daily_price_none.find(
                {"date": {"$gte": self.date_s, "$lte": self.date_e}},
                {"_id": 0, "date": 1, "code": 1, "close": 1, "high_limit": 1, "low_limit": 1},
                batch_size=1000000
            )
        )
        df_limit['date'] = pd.to_datetime(df_limit['date'])
        df_limit['limit'] = df_limit.apply(
            lambda x: x["close"] == x["high_limit"] or x["close"] == x["low_limit"],
            axis=1
        ).astype('int')
        
        # 创建并保存矩阵
        trade_status_matrix = df_info.pivot(index='date', columns='code', values='trade_status')
        riskwarning_matrix = df_info.pivot(index='date', columns='code', values='riskwarning')
        limit_matrix = df_limit.pivot(index='date', columns='code', values='limit')
        
        trade_status_matrix.to_pickle(matrix_paths['trade_status'])
        riskwarning_matrix.to_pickle(matrix_paths['riskwarning'])
        limit_matrix.to_pickle(matrix_paths['limit'])
        
        df_stocks = self.get_chg_wind()
        return df_stocks, trade_status_matrix, riskwarning_matrix, limit_matrix

    def generate_score_matrix(self, file_name: str) -> pd.DataFrame:
        """从CSV文件生成评分矩阵"""
        try:
            csv_file_path = os.path.join(self.data_folder, file_name)
            logger.debug(f"从本地文件 {csv_file_path} 加载数据...")
            
            df = pd.read_csv(csv_file_path)
            df['date'] = pd.to_datetime(df['date'])
            df.set_index('date', inplace=True)
            
            return df.pivot_table(
                index='date',
                columns='code',
                values='F1'
            )
            
        except Exception as e:
            logger.error(f"加载评分矩阵失败: {e}")
            raise

    def get_stocks_info_normal(self) -> pd.DataFrame:
        """获取普通股票信息"""
        t_info = self.client_U.basic_wind.w_basic_info
        df_info = pd.DataFrame(
            t_info.find(
                {"date": {"$gte": self.date_s, "$lte": self.date_e}},
                {"_id": 0, 'date': 1, 'code': 1, 'riskwarning': 1, 'trade_status': 1},
                batch_size=1000000
            )
        )
        return df_info.set_index(['date', 'code']).sort_index()

    def get_hold_num_per(self, percentage_1=0, percentage_2=0.03):
        """获取持仓数量（按百分比）"""
        df_info = self.get_stocks_info_normal()
        filtered_df = df_info[
            (df_info['trade_status'] == 1) & 
            (df_info['riskwarning'] == 0)
        ]
        
        daily_code_count = filtered_df.groupby('date').size()
        
        hold_s = (daily_code_count * percentage_1).apply(np.ceil).astype(int)
        hold_num = hold_s + (daily_code_count * percentage_2).apply(np.ceil).astype(int)
        
        return pd.DataFrame({
            'hold_s': hold_s,
            'hold_num': hold_num
        })

    def get_hold_num(self, hold_num=50, start_sorted=100, 
                    the_end_month=None, fixed_by_month=True):
        """获取持仓数量"""
        df = self._get_dynamic_nums()
        
        if the_end_month is None:
            the_end_count = df.iloc[-1]['count']
        else:
            the_end_count = df.loc[the_end_month]['count']

        df['hold_s'] = (df['count'] * (start_sorted / the_end_count)).apply(math.floor)
        df['hold_e'] = (df['count'] * ((start_sorted + hold_num) / the_end_count)).apply(math.floor)
        df['hold_num'] = df.hold_e - df.hold_s
        df['num_pre'] = df.hold_num.shift(-1)
        df['num_pre'] = df['hold_num']
        df.ffill(inplace=True)
        
        df['hold_num'] = df.apply(
            lambda dx: dx.hold_num if dx.hold_num <= dx.num_pre else dx.num_pre,
            axis=1
        ).astype(int)
        
        if the_end_month:
            df.loc[the_end_month:, 'hold_s'] = start_sorted
            if fixed_by_month:
                df.loc[the_end_month:, 'hold_num'] = hold_num
                
        return df[['hold_s', 'hold_num']]

    def _get_dynamic_nums(self):
        """获取动态数量"""
        # 获取每月第一个交易日
        df_dates = pd.DataFrame(
            self.client_U.economic.trade_dates.find(
                {"trade_date": {"$gte": self.date_s, "$lte": self.date_e}},
                {'_id': 0, 'trade_date': 1}
            )
        )
        df_dates['month'] = df_dates.trade_date.str[:7]
        df_dates = df_dates.loc[~df_dates.duplicated(subset=['month'], keep='first')]
        trade_first_m_day = df_dates.trade_date.tolist()

        # 计算每月第一天的有效股票数量
        pipeline = [
            {
                '$match': {
                    'date': {'$in': trade_first_m_day},
                    'industry': {'$nin': USELESS_INDUS},
                    'trade_days': {'$gte': 365}
                }
            },
            {
                '$group': {
                    '_id': '$date',
                    'count': {'$sum': 1}
                }
            }
        ]
        
        df_m_day_count = pd.DataFrame(
            list(self.client_U.basic_wind.w_basic_info.aggregate(pipeline)),
            columns=['_id', 'count']
        )
        df_m_day_count['month'] = df_m_day_count['_id'].str[:7]
        df_m_day_count.set_index('month', inplace=True)
        df_m_day_count.sort_index(inplace=True)
        
        return df_m_day_count