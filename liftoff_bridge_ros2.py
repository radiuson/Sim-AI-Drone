#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import math, socket, struct, time
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped, TwistStamped
from sensor_msgs.msg import Imu, Joy

def read_floats(buf, n):
    return struct.unpack("<" + "f"*n, buf[:4*n]), buf[4*n:]

def quat_multiply(q1, q2):
    x1,y1,z1,w1 = q1; x2,y2,z2,w2 = q2
    return (
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2,
        w1*w2 - x1*x2 - y1*y2 - z1*z2
    )

def quat_conjugate(q):
    x,y,z,w = q
    return (-x,-y,-z,w)

def R_to_quat(R):
    m00,m01,m02 = R[0]; m10,m11,m12 = R[1]; m20,m21,m22 = R[2]
    tr = m00 + m11 + m22
    if tr > 0:
        S = math.sqrt(tr+1.0) * 2
        w = 0.25 * S
        x = (m21 - m12) / S
        y = (m02 - m20) / S
        z = (m10 - m01) / S
    elif (m00 > m11) and (m00 > m22):
        S = math.sqrt(1.0 + m00 - m11 - m22) * 2
        w = (m21 - m12) / S
        x = 0.25 * S
        y = (m01 + m10) / S
        z = (m02 + m20) / S
    elif m11 > m22:
        S = math.sqrt(1.0 + m11 - m00 - m22) * 2
        w = (m02 - m20) / S
        x = (m01 + m10) / S
        y = 0.25 * S
        z = (m12 + m21) / S
    else:
        S = math.sqrt(1.0 + m22 - m00 - m11) * 2
        w = (m10 - m01) / S
        x = (m02 + m20) / S
        y = (m12 + m21) / S
        z = 0.25 * S
    return (x,y,z,w)

def quat_to_euler_xyz(qx,qy,qz,qw):
    """Returns (roll_x, pitch_y, yaw_z) in radians. Right-handed, XYZ order."""
    # Standard conversion, avoiding gimbal lock singularities
    # roll (x)
    sinr_cosp = 2*(qw*qx + qy*qz)
    cosr_cosp = 1 - 2*(qx*qx + qy*qy)
    roll = math.atan2(sinr_cosp, cosr_cosp)
    # pitch (y)
    sinp = 2*(qw*qy - qz*qx)
    if abs(sinp) >= 1:
        pitch = math.copysign(math.pi/2, sinp)  # 90°近奇异
    else:
        pitch = math.asin(sinp)
    # yaw (z)
    siny_cosp = 2*(qw*qz + qx*qy)
    cosy_cosp = 1 - 2*(qy*qy + qz*qz)
    yaw = math.atan2(siny_cosp, cosy_cosp)
    return roll, pitch, yaw

class LiftoffBridge(Node):
    def __init__(self):
        super().__init__('liftoff_bridge')

        # --- params ---
        self.declare_parameter('host', '127.0.0.1')
        self.declare_parameter('port', 30001)
        self.declare_parameter('R_rows', [1.0,0.0,0.0,  0.0,0.0,-1.0,  0.0,1.0,0.0])
        # Print rate (Hz), 0 = no printing
        self.declare_parameter('print_rate_hz', 2.0)

        host = self.get_parameter('host').get_parameter_value().string_value
        port = self.get_parameter('port').get_parameter_value().integer_value
        R_flat = list(self.get_parameter('R_rows').get_parameter_value().double_array_value)
        if len(R_flat) != 9:
            R_flat = [1.0,0.0,0.0,  0.0,0.0,-1.0,  0.0,1.0,0.0]
        R = [ R_flat[0:3], R_flat[3:6], R_flat[6:9] ]
        self.R = R
        self.qR = R_to_quat(R)
        self.qR_inv = quat_conjugate(self.qR)

        self.print_rate_hz = float(self.get_parameter('print_rate_hz').value)
        self._print_dt = 1.0/self.print_rate_hz if self.print_rate_hz > 0 else None
        self._last_print_t = time.time()

        # UDP
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.bind((host, port))
        self.sock.setblocking(False)

        # pubs
        self.pub_pose  = self.create_publisher(PoseStamped,  '/liftoff/pose', 10)
        self.pub_twist = self.create_publisher(TwistStamped, '/liftoff/twist', 10)
        self.pub_imu   = self.create_publisher(Imu,          '/liftoff/imu',   10)
        self.pub_rc    = self.create_publisher(Joy,          '/liftoff/rc',    10)

        # State cache (for printing)
        self._last_p = (0.0,0.0,0.0)
        self._last_v = (0.0,0.0,0.0)
        self._last_q = (0.0,0.0,0.0,1.0)

        # For computing linear acceleration
        self._last_t = None          # Last timestamp (Liftoff timestamp)
        self._last_a = (0.0,0.0,0.0) # Last acceleration

        # 500Hz polling loop
        self.timer = self.create_timer(1/500.0, self.loop)

    def apply_R(self, v):
        R = self.R
        return (
            R[0][0]*v[0] + R[0][1]*v[1] + R[0][2]*v[2],
            R[1][0]*v[0] + R[1][1]*v[1] + R[1][2]*v[2],
            R[2][0]*v[0] + R[2][1]*v[1] + R[2][2]*v[2],
        )

    def _maybe_print(self):
        if self._print_dt is None:
            return
        now = time.time()
        if now - self._last_print_t >= self._print_dt:
            px,py,pz = self._last_p
            vx,vy,vz = self._last_v
            qx,qy,qz,qw = self._last_q
            roll, pitch, yaw = quat_to_euler_xyz(qx,qy,qz,qw)
            speed = math.sqrt(vx*vx + vy*vy + vz*vz)
            # Terminal output: pitch angle (degrees), speed (m/s), altitude (m)
            self.get_logger().info(
                f"pitch={math.degrees(pitch):6.2f} deg | speed={speed:6.2f} m/s | alt(z)={pz:6.2f} m"
            )
            self._last_print_t = now

    def loop(self):
        try:
            data, _ = self.sock.recvfrom(2048)
        except BlockingIOError:
            self._maybe_print()
            return

        # Validate minimum packet length (1+3+4+3+3+4+2 = 20 floats = 80 bytes)
        if len(data) < 80:
            self.get_logger().warn(f'UDP packet too short: {len(data)} bytes < 80 bytes')
            return

        try:
            ts,   data = read_floats(data, 1)
            pos,  data = read_floats(data, 3)
            att,  data = read_floats(data, 4)
            vel,  data = read_floats(data, 3)
            gyro, data = read_floats(data, 3)  # deg/s
            _in,  data = read_floats(data, 4)
            bat,  data = read_floats(data, 2)
            n_motors = int(data[0]) if len(data)>=1 else 0
            data = data[1:] if len(data)>=1 else data
            if n_motors>0:
                motors, data = read_floats(data, int(n_motors))
            else:
                motors = []
        except (struct.error, IndexError) as e:
            self.get_logger().warn(f'UDP packet parsing failed: {e}')
            return

        # Position/velocity mapping
        p_ros = self.apply_R(pos)
        v_ros = self.apply_R(vel)

        # Attitude: q_ros = qR * q_l * qR^{-1}
        ql = (att[0], att[1], att[2], att[3])  # (x,y,z,w)
        q_ros = quat_multiply(self.qR, quat_multiply(ql, self.qR_inv))

        # Angular velocity: Liftoff (pitch,roll,yaw) deg/s -> (roll,pitch,yaw) -> R -> rad/s
        gyro_vec = (gyro[1], gyro[0], gyro[2])
        w_ros = self.apply_R(gyro_vec)
        w_ros = tuple(w*math.pi/180.0 for w in w_ros)

        stamp = self.get_clock().now().to_msg()

        # Publish messages
        msgP = PoseStamped()
        msgP.header.frame_id = 'map'
        msgP.header.stamp = stamp
        msgP.pose.position.x, msgP.pose.position.y, msgP.pose.position.z = p_ros
        msgP.pose.orientation.x, msgP.pose.orientation.y, msgP.pose.orientation.z, msgP.pose.orientation.w = q_ros
        self.pub_pose.publish(msgP)

        msgT = TwistStamped()
        msgT.header.frame_id = 'map'  # Velocity expressed in world frame
        msgT.header.stamp = stamp
        msgT.twist.linear.x, msgT.twist.linear.y, msgT.twist.linear.z = v_ros
        msgT.twist.angular.x, msgT.twist.angular.y, msgT.twist.angular.z = w_ros
        self.pub_twist.publish(msgT)

        msgI = Imu()
        msgI.header.frame_id = 'base_link'
        msgI.header.stamp = stamp
        msgI.orientation.x, msgI.orientation.y, msgI.orientation.z, msgI.orientation.w = q_ros
        msgI.angular_velocity.x, msgI.angular_velocity.y, msgI.angular_velocity.z = w_ros

        # Set covariance matrices (orientation, angular_velocity, linear_acceleration)
        # For simulation data, use small covariance values to indicate high precision
        # If covariance is unknown, can set [0]=0; here we use typical simulation values
        msgI.orientation_covariance = [1e-6, 0.0, 0.0,
                                        0.0, 1e-6, 0.0,
                                        0.0, 0.0, 1e-6]
        msgI.angular_velocity_covariance = [1e-4, 0.0, 0.0,
                                             0.0, 1e-4, 0.0,
                                             0.0, 0.0, 1e-4]
        msgI.linear_acceleration_covariance = [1e-3, 0.0, 0.0,
                                                0.0, 1e-3, 0.0,
                                                0.0, 0.0, 1e-3]

        # --- Acceleration computation (using Liftoff timestamp) ---
        t_now = ts[0]  # Use Liftoff simulation timestamp instead of system time
        if self._last_t is not None:
            dt = t_now - self._last_t
            if dt > 1e-4:  # Avoid division by zero
                ax = (v_ros[0] - self._last_v[0]) / dt
                ay = (v_ros[1] - self._last_v[1]) / dt
                az = (v_ros[2] - self._last_v[2]) / dt
                self._last_a = (ax, ay, az)
        msgI.linear_acceleration.x, msgI.linear_acceleration.y, msgI.linear_acceleration.z = self._last_a
        self._last_v = v_ros
        self._last_t = t_now

        # Publish IMU
        self.pub_imu.publish(msgI)


        msgJ = Joy()
        msgJ.header.stamp = stamp
        msgJ.axes = list(_in)  # [thr,yaw,pitch,roll]
        self.pub_rc.publish(msgJ)

        # Update cache and throttled printing
        self._last_p = p_ros
        self._last_v = v_ros
        self._last_q = q_ros
        self._maybe_print()

def main():
    rclpy.init()
    node = LiftoffBridge()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
