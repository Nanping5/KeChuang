import sys
import time
from connect import ConnectRobot
import re

try:
    import msvcrt  # 仅Windows
except ImportError:
    print("本脚本仅支持Windows平台。")
    sys.exit(1)

def main():
    dashboard, move, feed = ConnectRobot()
    dashboard.SpeedJ(30)
    dashboard.SpeedL(30)
    print("使用方向键和自定义按键控制机械臂，按q退出。")
    print("↑: y+1, ↓: y-1, ←: x-1, →: x+1, u: r+1, j: r-1")
    step_xy = 1
    step_z = 1
    step_r = 1

    while True:
        if msvcrt.kbhit():
            key = msvcrt.getch()
            if key == b'q':
                print("退出控制。")
                break
            elif key == b'\xe0':
                key2 = msvcrt.getch()
                # 获取当前位置
                pose_str = dashboard.GetPose()
                try:
                    match = re.search(r"\{([^}]*)\}", pose_str)
                    if not match:
                        raise ValueError("未找到花括号内容")
                    pose_data = match.group(1)
                    x, y, z, r = [float(v) for v in pose_data.split(',')[:4]]
                except Exception as e:
                    print(f"GetPose解析失败: {pose_str}")
                    continue
                # 先判断当前位置是否在限位内（只考虑x/y）
                if not (200 < x < 400 and -200 < y < 86):
                    # 只允许回安全区方向移动
                    allowed = False
                    if key2 == b'H' and y < -200:
                        print("上 (y+1)")
                        move.RelMovL(0, step_xy, 0, 0)
                        allowed = True
                    elif key2 == b'P' and y > 86:
                        print("下 (y-1)")
                        move.RelMovL(0, -step_xy, 0, 0)
                        allowed = True
                    elif key2 == b'K' and x > 400:
                        print("左 (x-1)")
                        move.RelMovL(-step_xy, 0, 0, 0)
                        allowed = True
                    elif key2 == b'M' and x < 200:
                        print("右 (x+1)")
                        move.RelMovL(step_xy, 0, 0, 0)
                        allowed = True
                    elif key2 == b'H' and y >= -200 and y < 86:
                        print("上 (y+1)")
                        move.RelMovL(0, step_xy, 0, 0)
                        allowed = True
                    elif key2 == b'P' and y > -200 and y <= 86:
                        print("下 (y-1)")
                        move.RelMovL(0, -step_xy, 0, 0)
                        allowed = True
                    elif key2 == b'K' and x > 200 and x <= 400:
                        print("左 (x-1)")
                        move.RelMovL(-step_xy, 0, 0, 0)
                        allowed = True
                    elif key2 == b'M' and x >= 200 and x < 400:
                        print("右 (x+1)")
                        move.RelMovL(step_xy, 0, 0, 0)
                        allowed = True
                    if not allowed:
                        print("当前位置已超限，只允许回安全区方向移动！")
                    continue
                if key2 == b'H':  # 上
                    new_y = y + step_xy
                    if 200 < x < 400 and -200 < new_y < 86:
                        print("上 (y+1)")
                        move.RelMovL(0, step_xy, 0, 0)
                    else:
                        print("超出限位，禁止移动！")
                elif key2 == b'P':  # 下
                    new_y = y - step_xy
                    if 200 < x < 400 and -200 < new_y < 86:
                        print("下 (y-1)")
                        move.RelMovL(0, -step_xy, 0, 0)
                    else:
                        print("超出限位，禁止移动！")
                elif key2 == b'K':  # 左
                    new_x = x - step_xy
                    if 200 < new_x < 400 and -200 < y < 86:
                        print("左 (x-1)")
                        move.RelMovL(-step_xy, 0, 0, 0)
                    else:
                        print("超出限位，禁止移动！")
                elif key2 == b'M':  # 右
                    new_x = x + step_xy
                    if 200 < new_x < 400 and -200 < y < 86:
                        print("右 (x+1)")
                        move.RelMovL(step_xy, 0, 0, 0)
                    else:
                        print("超出限位，禁止移动！")
            elif key == b'u':
                # r轴不做限位
                print("r+1")
                move.RelMovJ(0, 0, 0, step_r)
            elif key == b'j':
                print("r-1")
                move.RelMovJ(0, 0, 0, -step_r)
            # 检查移动过程中是否有q打断
            for _ in range(10):  # 约0.1秒内轮询
                if msvcrt.kbhit() and msvcrt.getch() == b'q':
                    print("移动中检测到退出指令，已打断。")
                    return
                time.sleep(0.01)
        else:
            time.sleep(0.01)

if __name__ == "__main__":
    main() 