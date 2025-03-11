"""
MongoDB数据库连接客户端
"""
import pymongo
from urllib.parse import quote_plus
from loguru import logger
from config.settings import MONGO_CONFIG

def get_client(c_from='local'):
    """获取MongoDB客户端连接"""
    client_dict = {
        'local': 'localhost:27017',
        'dev': '192.168.1.78:27017',
    }
    
    client_name = client_dict.get(c_from)
    if not client_name:
        raise ValueError(f'Invalid database target: {c_from}')
        
    return pymongo.MongoClient(f"mongodb://{client_name}")

def get_client_U(m='r'):
    """获取带用户认证的MongoDB客户端连接"""
    auth_config = {
        'r': ('Tom', 'tom'),
        'rw': ('Amy', 'amy'), 
        'Neo': ('Neo', 'neox')
    }
    
    user, pwd = auth_config.get(m, ('Tom', 'tom'))
    if m not in auth_config:
        logger.warning(f'Invalid auth type {m}, using read-only access')
        
    return pymongo.MongoClient(
        f"mongodb://{quote_plus(user)}:{quote_plus(pwd)}@192.168.1.99:29900/"
    ) 