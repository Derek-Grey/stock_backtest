"""
项目配置文件
"""
from pathlib import Path

# 基础路径配置
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
LOG_DIR = BASE_DIR / "logs" 
OUTPUT_DIR = BASE_DIR / "output"

# MongoDB连接配置
MONGO_CONFIG = {
    'DEV': 'mongodb://localhost:27017/',
    'NEO': 'mongodb://192.168.1.77:27017/',
    'SERVER': 'mongodb://192.168.1.76:27017/'
}

# 数据库名称配置
DB_NAMES = {
    'WIND': 'basic_wind',
    'JQ': 'basic_jq',
    'ECO': 'economic',
    'MINUTE': 'minute_jq'
}

# 不参与RANK的行业列表
USELESS_INDUS = [
    "证券、期货业", "银行业", "货币金融服务", "其他金融业",
    "资本市场服务", "保险业", "燃气生产和供应业",
    # ... 其他行业保持不变
]

# 策略回测配置
STRATEGY_CONFIG = {
    "alpha_first": ("bt_alpha_first", 0, 50),
    'yk_first': ('bt_yk_first', 100, 50),
}

END_MONTH = '2022-07' 