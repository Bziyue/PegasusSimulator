"""
| File: isaac_imu.py
| Description: Isaac Sim 原生 IMU 传感器封装，无噪声版本
| 用于与 Pegasus 插件内的 IMU 进行对比
"""
__all__ = ["IsaacIMU"]

import numpy as np
from scipy.spatial.transform import Rotation

from pegasus.simulator.logic.state import State
from pegasus.simulator.logic.sensors.sensor import Sensor
from pegasus.simulator.logic.interface.pegasus_interface import PegasusInterface

# Imports the python bindings to interact with IMU sensor
import omni.kit.commands
from pxr import Gf, UsdGeom, Sdf
from omni.usd import get_stage_next_free_path

# Isaac Sim IMU sensor interface
from isaacsim.sensors.physics import _sensor as sensor_interface


class IsaacIMU(Sensor):
    """
    Isaac Sim 原生 IMU 传感器封装类
    
    该传感器直接使用 Isaac Sim 的物理引擎计算 IMU 数据，
    可以配置为无噪声模式，用于获取理想的 IMU 测量值。
    
    注意：继承自 Sensor（而非 GraphicalSensor），确保每次物理步进都更新。
    """

    def __init__(self, imu_name: str, config: dict = {}):
        """
        初始化 Isaac Sim 原生 IMU 传感器

        Args:
            imu_name: IMU 传感器名称
            config: 配置字典，支持以下参数:
                - position: [x, y, z] 安装位置 (相对于机体坐标系)，默认 [0, 0, 0]
                - orientation: [roll, pitch, yaw] 安装姿态 (度)，默认 [0, 0, 0]
                - update_rate: 更新频率 (Hz)，默认 250.0
        """
        # 初始化父类 - 使用 "IsaacIMU" 作为传感器类型
        super().__init__(sensor_type="IsaacIMU", update_rate=config.get("update_rate", 250.0))

        # 传感器名称
        self._imu_name = imu_name
        self._stage_prim_path = ""

        # 安装位置和姿态
        self._position = np.array(config.get("position", [0.0, 0.0, 0.0]))
        self._orientation = Rotation.from_euler(
            "ZYX", 
            config.get("orientation", [0.0, 0.0, 0.0]), 
            degrees=True
        ).as_quat()

        # Isaac Sim 传感器接口
        self._sensor_interface = None
        self._imu_sensor_handle = None

        # 传感器状态
        self._state = {
            "angular_velocity": np.array([0.0, 0.0, 0.0]),
            "linear_acceleration": np.array([0.0, 0.0, 0.0]),
            "orientation": np.array([0.0, 0.0, 0.0, 1.0]),  # quaternion [x, y, z, w]
        }

    def initialize(self, vehicle, latitude=None, longitude=None, altitude=None):
        """
        初始化 IMU 传感器，将其附加到无人机上
        
        Args:
            vehicle: 飞行器对象
            latitude: 纬度（未使用，保持接口兼容）
            longitude: 经度（未使用，保持接口兼容）
            altitude: 高度（未使用，保持接口兼容）
        """
        # 保存 vehicle 引用（Sensor 基类需要）
        self._vehicle = vehicle

        # 获取传感器接口
        self._sensor_interface = sensor_interface.acquire_imu_sensor_interface()

        # 生成唯一的 prim 路径
        self._stage_prim_path = get_stage_next_free_path(
            PegasusInterface().world.stage,
            self._vehicle.prim_path + "/body/" + self._imu_name,
            False
        )

        # 更新实际创建的传感器名称
        self._imu_name = self._stage_prim_path.rpartition("/")[-1]

        # 使用 Isaac Sim 命令创建 IMU 传感器
        result, self._imu_prim = omni.kit.commands.execute(
            "IsaacSensorCreateImuSensor",
            path="/" + self._imu_name,
            parent=self._vehicle.prim_path + "/body",
            sensor_period=-1,  # 使用仿真步长
            translation=Gf.Vec3d(self._position[0], self._position[1], self._position[2]),
            orientation=Gf.Quatd(self._orientation[3], self._orientation[0], self._orientation[1], self._orientation[2]),
        )

        if result:
            self._stage_prim_path = str(self._imu_prim.GetPath())
            print(f"[IsaacIMU] 传感器创建成功: {self._stage_prim_path}")
        else:
            print(f"[IsaacIMU] 传感器创建失败!")

    @property
    def state(self):
        """传感器状态"""
        return self._state

    @Sensor.update_at_rate
    def update(self, state: State, dt: float):
        """
        更新 IMU 传感器数据

        Args:
            state: 当前飞行器状态
            dt: 时间步长

        Returns:
            包含 IMU 数据的字典
        """
        if self._sensor_interface is None:
            return self._state

        # 从 Isaac Sim 传感器接口读取 IMU 数据
        imu_reading = self._sensor_interface.get_sensor_reading(self._stage_prim_path)

        if imu_reading.is_valid:
            # 更新角速度 (rad/s)
            self._state["angular_velocity"] = np.array([
                imu_reading.ang_vel_x,
                imu_reading.ang_vel_y,
                imu_reading.ang_vel_z
            ])

            # 更新线性加速度 (m/s^2)，包含重力
            self._state["linear_acceleration"] = np.array([
                imu_reading.lin_acc_x,
                imu_reading.lin_acc_y,
                imu_reading.lin_acc_z
            ])

            # 更新姿态四元数
            self._state["orientation"] = np.array([
                imu_reading.orientation.x,
                imu_reading.orientation.y,
                imu_reading.orientation.z,
                imu_reading.orientation.w
            ])

        return self._state
