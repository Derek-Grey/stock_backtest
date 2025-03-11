"""
通用辅助函数
"""
import pandas as pd
from loguru import logger

def trans_str_to_float64(df: pd.DataFrame, exp_cols: list = None, trans_cols: list = None) -> pd.DataFrame:
    """将DataFrame中的字符串列转换为float64类型"""
    if trans_cols is None and exp_cols is None:
        trans_cols = df.columns
    if exp_cols is not None:
        trans_cols = list(set(df.columns) - set(exp_cols))
    df[trans_cols] = df[trans_cols].astype('float64')
    return df

# 添加其他辅助函数... 