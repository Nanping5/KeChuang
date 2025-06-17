import cv2
import numpy as np
import xml.etree.ElementTree as ET
@staticmethod
def parse_xml_points(xml_path, point_list_name):
    """
    解析XML文件中的点坐标数据
    :param xml_path: XML文件路径
    :param point_list_name: 点列表名称（ImagePointLst或WorldPointLst）
    :return: 包含点坐标的numpy数组
    """
    tree = ET.parse(xml_path)
    root = tree.getroot()
    print(root)
    points = []
    for param in root.findall(f'.//CalibPointFListParam[@ParamName="{point_list_name}"]/PointF'):
        x = float(param.find('X').text)
        y = float(param.find('Y').text)
        points.append([x, y])
    
    return np.array(points, dtype=np.float32)
@staticmethod
def getAffineMat(points_camera, points_robot):
    """
    取得相机坐标转换到机器坐标的仿射矩阵
    :param points_camera: 相机坐标点集
    :param points_robot: 机器人坐标点集
    :return: 仿射变换矩阵
    """
    # 确保两个点集的数量级不要差距过大，否则会输出None
    m, _ = cv2.estimateAffine2D(points_camera, points_robot)
    return m

def handEyeCalibration(xml_path):
    """
    相机坐标通过仿射矩阵变换取得机器坐标
    : param xml_path: XML文件路径
    :param x_camera: 摄像头的x坐标
    :param y_camera: 摄像头的y坐标
    :return: 转换后的机器人坐标(x, y)
    """
    # 从XML文件解析标定点
    camera_points = parse_xml_points(xml_path, 'ImagePointLst')
    robot_points = parse_xml_points(xml_path, 'WorldPointLst')
    # 获取仿射变换矩阵
    aff_m = getAffineMat(camera_points, robot_points)
    return aff_m
def useAffineMat(aff_m, x_camera, y_camera):
    # 应用仿射变换
    robot_x = (aff_m[0][0] * x_camera) + (aff_m[0][1] * y_camera) + aff_m[0][2]
    robot_y = (aff_m[1][0] * x_camera) + (aff_m[1][1] * y_camera) + aff_m[1][2]
    return robot_x, robot_y

# 测试代码
aff_m=handEyeCalibration(r'biaodingXML\250617.xml')
test_point = useAffineMat(  aff_m,1137, 724)  # 使用第一个标定点测试
print(f"转换后的机器人坐标: {test_point}")

