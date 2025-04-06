import open3d as o3d

def export_point_cloud(points, filename):
    """
    导出点云到 PLY 文件
    Args:
        points: [N, 3] numpy 数组，点的坐标
        filename: 导出的文件名
    """
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)  # 设置点的位置
    o3d.io.write_point_cloud(filename, pcd)  # 保存点云为 PLY 文件
