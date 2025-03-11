"""
回测主程序
用于运行和测试回测策略
"""
import sys
from pathlib import Path
from loguru import logger
import pandas as pd

# 添加项目根目录到系统路径
ROOT_DIR = Path(__file__).resolve().parent.parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

from src.data.load_data import LoadData
from src.backtest.backtest import Backtest
from config.settings import DATA_DIR, OUTPUT_DIR

def process_data(data_loader):
    """
    处理和对齐数据
    
    Args:
        data_loader: LoadData实例
        
    Returns:
        tuple: (aligned_stocks_matrix, aligned_limit_matrix,
                aligned_riskwarning_matrix, aligned_trade_status_matrix,
                score_matrix)
    """
    try:
        # 获取开始和结束日期
        start_date = pd.to_datetime(data_loader.date_s)
        end_date = pd.to_datetime(data_loader.date_e)
        logger.debug(f"处理数据: 从 {start_date} 到 {end_date}")

        # 创建数据目录
        data_folder = Path(data_loader.data_folder)
        data_folder.mkdir(parents=True, exist_ok=True)

        # 获取基础数据
        df_stocks, trade_status_matrix, riskwarning_matrix, limit_matrix = data_loader.get_stocks_info()
        logger.debug(f"获取到的矩阵形状: stocks={df_stocks.shape}, "
                    f"trade_status={trade_status_matrix.shape}, "
                    f"risk_warning={riskwarning_matrix.shape}, "
                    f"limit={limit_matrix.shape}")
        
        # 生成评分矩阵
        score_matrix = data_loader.generate_score_matrix('stra_V3_11.csv')
        logger.debug(f"评分矩阵形状: {score_matrix.shape}")

        # 对齐数据
        aligned_stocks_matrix = align_and_fill_matrix(df_stocks, score_matrix)
        aligned_limit_matrix = align_and_fill_matrix(limit_matrix, score_matrix)
        aligned_riskwarning_matrix = align_and_fill_matrix(riskwarning_matrix, score_matrix)
        aligned_trade_status_matrix = align_and_fill_matrix(trade_status_matrix, score_matrix)

        # 保存对齐后的矩阵
        for matrix, filename in [
            (aligned_stocks_matrix, 'aligned_stocks_matrix.csv'),
            (aligned_limit_matrix, 'aligned_limit_matrix.csv'),
            (aligned_riskwarning_matrix, 'aligned_riskwarning_matrix.csv'),
            (aligned_trade_status_matrix, 'aligned_trade_status_matrix.csv'),
            (score_matrix, 'aligned_score_matrix.csv')
        ]:
            matrix.to_csv(data_folder / filename)
            logger.debug(f"保存矩阵到 {filename}")

        return (aligned_stocks_matrix, aligned_limit_matrix,
                aligned_riskwarning_matrix, aligned_trade_status_matrix,
                score_matrix)

    except Exception as e:
        logger.error(f"数据处理失败: {str(e)}")
        logger.exception("详细错误信息:")
        return None

def align_and_fill_matrix(target_matrix: pd.DataFrame, reference_matrix: pd.DataFrame) -> pd.DataFrame:
    """
    将目标矩阵与参考矩阵的列对齐，并用0填充缺失值
    """
    try:
        aligned_matrix = target_matrix.reindex(columns=reference_matrix.columns, fill_value=0)
        return aligned_matrix
    except Exception as e:
        logger.error(f"对齐矩阵失败: {e}")
        raise

def run_strategy(backtest, strategy_name, hold_count, rebalance_frequency, df_mv=None):
    """
    运行回测策略并保存结果
    
    Args:
        backtest: Backtest实例
        strategy_name: 策略名称
        hold_count: 固定持仓数量
        rebalance_frequency: 再平衡频率（天数）
        df_mv: 包含每月持仓数量的DataFrame（仅用于动态策略）
    """
    logger.info(f"运行{strategy_name}策略...")
    try:
        if strategy_name == "fixed":
            results = backtest.run_fixed_strategy(
                hold_count=hold_count,
                rebalance_frequency=rebalance_frequency,
                strategy_name=strategy_name
            )
        elif strategy_name == "dynamic":
            results = backtest.run_dynamic_strategy(
                rebalance_frequency=rebalance_frequency,
                df_mv=df_mv,
                start_sorted=100
            )
        
        backtest.plot_results(results, strategy_name)
        save_results(results, strategy_name, backtest.output_dir)
        logger.info(f"{strategy_name}策略完成")
        return results
    
    except Exception as e:
        logger.error(f"{strategy_name}策略执行失败: {e}")
        return None

def save_results(results, strategy_name, output_directory):
    """保存回测结果到CSV文件"""
    if results is not None:
        output_path = Path(output_directory) / f'{strategy_name}_results.csv'
        results.to_csv(output_path)
        logger.info(f"{strategy_name}结果已保存到 {output_path}")

def main(start_date="2010-08-02", end_date="2020-07-31", 
         hold_count=50, rebalance_frequency=1,
         run_fixed=True, run_dynamic=True,
         position_type="固定数量",  # 新增参数
         start_percentage=0.01,     # 新增参数
         end_percentage=0.03):      # 新增参数
    """
    主函数，执行回测策略
    
    Args:
        start_date: 回测开始日期，格式："YYYY-MM-DD"
        end_date: 回测结束日期，格式："YYYY-MM-DD"
        hold_count: 持仓数量，默认50只股票
        rebalance_frequency: 再平衡频率（天数），默认每天再平衡
        run_fixed: 是否运行固定持仓策略
        run_dynamic: 是否运行动态持仓策略
        position_type: 持仓方式，"固定数量"或"动态百分比"
        start_percentage: 起始持仓百分比（动态百分比模式使用）
        end_percentage: 结束持仓百分比（动态百分比模式使用）
    """
    try:
        logger.info(f"开始回测 - 时间范围: {start_date} 至 {end_date}")
        
        # 初始化数据加载器
        data_loader = LoadData(
            date_s=start_date,
            date_e=end_date,
            data_folder=DATA_DIR
        )
        
        # 处理数据
        matrices = process_data(data_loader)
        if matrices is None:
            raise ValueError("数据处理失败，无法获取所需矩阵")
            
        # 初始化回测实例
        backtest = Backtest(*matrices, output_dir=OUTPUT_DIR)
        
        results = {}
        
        # 运行固定持仓策略
        if run_fixed:
            if position_type == "固定数量":
                results['fixed'] = run_strategy(
                    backtest, "fixed", hold_count, rebalance_frequency
                )
            else:  # 动态百分比
                df_mv = data_loader.get_hold_num_per(start_percentage, end_percentage)
                results['fixed'] = run_strategy(
                    backtest, "fixed", hold_count, rebalance_frequency, df_mv
                )
        
        # 运行动态持仓策略
        if run_dynamic:
            if position_type == "固定数量":
                df_mv = data_loader.get_hold_num()
            else:  # 动态百分比
                df_mv = data_loader.get_hold_num_per(start_percentage, end_percentage)
                
            results['dynamic'] = run_strategy(
                backtest, "dynamic", hold_count, rebalance_frequency, df_mv
            )
        
        # 合并结果
        if all(result is not None for result in results.values()):
            combined_results = pd.concat(list(results.values()), axis=0)
            combined_results.to_csv(Path(OUTPUT_DIR) / 'combined_results.csv')
            logger.info("固定和动态持仓结果已合并并保存")
        
        return results.get('fixed'), results.get('dynamic')
    
    except Exception as e:
        logger.error(f"回测执行失败: {e}")
        raise

if __name__ == "__main__":
    # 配置日志
    logger.add(
        Path(ROOT_DIR) / "logs/backtest_{time}.log",
        rotation="1 day",
        retention="7 days",
        level="INFO"
    )
    
    # 设置回测参数
    params = {
        'start_date': "2015-01-05",
        'end_date': "2023-12-29",
        'hold_count': 50,
        'rebalance_frequency': 1,
        'run_fixed': True,
        'run_dynamic': True,
        'position_type': "固定数量",
        'start_percentage': 0.01,
        'end_percentage': 0.03
    }
    
    # 执行回测
    try:
        fixed_results, dynamic_results = main(**params)
        
        # 输出回测结果摘要
        if fixed_results is not None:
            logger.info("\n=== 固定持仓策略结果摘要 ===")
            cumulative_return = (1 + fixed_results['daily_return']).cumprod().iloc[-1] - 1
            logger.info(f"累计收益率: {cumulative_return:.2%}")
            
        if dynamic_results is not None:
            logger.info("\n=== 动态持仓策略结果摘要 ===")
            cumulative_return = (1 + dynamic_results['daily_return']).cumprod().iloc[-1] - 1
            logger.info(f"累计收益率: {cumulative_return:.2%}")
            
    except Exception as e:
        logger.error(f"程序执行失败: {e}") 