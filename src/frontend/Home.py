"""
量化策略回测系统主页
"""
import streamlit as st
from pathlib import Path
import sys
import pymongo

# 设置页面配置
st.set_page_config(
    page_title="量化策略回测系统",
    page_icon="📈",
    layout="wide"
)

# 添加项目根目录到系统路径
ROOT_DIR = Path(__file__).resolve().parent.parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

# Add custom CSS
st.markdown(
    """
    <style>
    .reportview-container {
        background: #FFFACD; /* 浅黄色背景 */
    }
    .sidebar .sidebar-content {
        background: #FFFFFF; /* 白色背景 */
    }
    .stButton>button {
        color: #FFFFFF;
        background-color: #FFD700; /* 金黄色按钮 */
        border-radius: 5px;
        border: none;
        padding: 10px 20px;
        font-size: 16px;
    }
    .stTitle {
        color: #333333; /* 深灰色标题 */
        font-weight: bold;
    }
    .stMarkdown h3 {
        color: #FFD700; /* 金黄色小标题 */
    }
    </style>
    """,
    unsafe_allow_html=True
)

def main():
    # 修改侧边栏页面名称
    st.sidebar.markdown(
        """
        <style>
        [data-testid="stSidebarNav"] li div a {
            margin-left: 1rem;
            padding: 1rem;
            width: 100%;
            font-weight: normal;
        }
        [data-testid="stSidebarNav"] div button p {
            font-weight: normal;
        }
        </style>
        """,
        unsafe_allow_html=True
    )
    
    # 设置页面标题和内容
    st.title("📈 量化策略回测系统")
    
    st.markdown("""
    ### 👋 欢迎使用量化策略回测系统
    
    本系统提供以下功能：
    
    * 📊 **策略回测**: 运行固定持仓和动态持仓策略的回测
    * 📈 **数据查看**: 查看和分析历史数据
    * 📝 **回测记录**: 查看历史回测结果及分析
    
    请从左侧边栏选择功能开始使用。
    """)
    
    # 显示系统信息
    st.sidebar.markdown("### 系统信息")
    st.sidebar.info(
        f"""
        - 数据更新时间: {Path(ROOT_DIR)/'data'/'last_update.txt'}
        - 系统版本: v1.0.0
        """
    )

if __name__ == "__main__":
    main() 