from dobot_api import DobotApi,DobotApiDashboard,DobotApiMove

def ConnectRobot():
    try:
        ip="192.168.1.6"
        dashboardPort = 29999
        movePort = 30003
        feedPort = 30004
        print("ing。。。")
        dashboard = DobotApiDashboard(ip,dashboardPort)
        move = DobotApiMove(ip,movePort)
        feed = DobotApi(ip,feedPort)
        dashboard.ClearError()
        dashboard.EnableRobot()
        print("success")
        return dashboard,move,feed
    except Exception as e:
        print("fail")
        raise e

