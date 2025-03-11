"""
启动量化策略回测系统
"""
import subprocess
import socket
from pathlib import Path

def find_free_port(start_port=8501, max_port=8999):
    """查找可用的端口号"""
    for port in range(start_port, max_port + 1):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            try:
                s.bind(('', port))
                return port
            except OSError:
                continue
    raise RuntimeError(f"在{start_port}-{max_port}范围内没有找到可用端口")

def main():
    # 获取前端应用入口文件路径
    frontend_path = Path(__file__).resolve().parent / "src" / "frontend" / "Home.py"
    
    # 查找可用端口
    try:
        port = find_free_port()
        print(f"使用端口: {port}")
    except RuntimeError as e:
        print(f"错误: {e}")
        return

    # 设置 Streamlit 命令行参数
    cmd = [
        "streamlit",
        "run",
        str(frontend_path),
        f"--server.port={port}",
        "--server.address=0.0.0.0",
        "--browser.serverAddress=localhost",
        "--theme.primaryColor=#FF4B4B",
        "--theme.backgroundColor=#FFFFFF",
        "--theme.secondaryBackgroundColor=#F0F2F6",
        "--theme.textColor=#262730"
    ]

    # 使用subprocess启动streamlit
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        print(f"启动失败: {e}")
    except KeyboardInterrupt:
        print("\n程序已终止")

if __name__ == "__main__":
    main()