# ROS 2 핵심 개념 (ROS 2 Core Concepts)
*Robot Operating System 2 - Fundamental Architecture and Communication Patterns*

ROS 2는 로봇 애플리케이션을 구성하는 여러 프로그램(노드)들이 서로 통신할 수 있도록 지원하는 프레임워크입니다. 이러한 통신은 토픽, 서비스, 액션이라는 세 가지 주요 메커니즘을 통해 이루어집니다. 이 문서는 ROS 2의 가장 기본적인 구성 요소인 노드와 세 가지 통신 방식에 대해 설명합니다.

## 1. 노드 (Nodes)

**노드(Node)**는 ROS 2 시스템에서 실행되는 가장 작은 단위의 프로세스입니다. 각 노드는 특정한 단일 목적을 수행하도록 설계됩니다. 예를 들어, 로봇 시스템은 다음과 같은 여러 노드로 구성될 수 있습니다.

- 카메라로부터 이미지를 받아오는 노드
- 라이다(LiDAR) 센서 데이터를 처리하는 노드
- 바퀴 모터를 제어하는 노드
- 현재 로봇의 위치를 추정하는 노드
- 로봇의 이동 경로를 계획하는 노드

이렇게 기능을 잘게 나누어 모듈화하면 시스템의 특정 부분을 독립적으로 개발하고 테스트하며 재사용하기 용이해집니다. 노드들은 ROS 2의 통신 시스템을 통해 서로 데이터를 주고받으며 하나의 통합된 로봇 애플리케이션으로 동작합니다.

## 2. ROS 2 통신 방식 (Communication Types)

ROS 2의 통신은 DDS(Data Distribution Service)를 기반으로 하며, 크게 세 가지 방식으로 나뉩니다.

![ROS 2 Communication](https-i-imgur-com-y91234-png)
*(실제 이미지 대신 설명을 위한 예시입니다. ROS 2의 통신 방식을 시각적으로 표현)*

### 2.1. 토픽 (Topics)

**토픽(Topic)**은 **지속적인 데이터 스트림**을 비동기적으로 주고받는 **일대다(one-to-many)** 통신 방식입니다. 데이터를 발행(Publish)하는 노드(Publisher)와 데이터를 구독(Subscribe)하는 노드(Subscriber)가 존재합니다.

- **특징:**
  - **비동기(Asynchronous):** 발행자는 구독자가 데이터를 받았는지 확인하지 않고 계속해서 메시지를 보냅니다.
  - **다대다(Many-to-Many):** 하나의 토픽에 여러 발행자와 여러 구독자가 연결될 수 있습니다.
  - **메시지(Message):** 토픽을 통해 주고받는 데이터의 구조를 `.msg` 파일에 정의합니다. (예: `geometry_msgs/msg/Twist`)

- **사용 예시:**
  - `/image_raw`: 카메라 드라이버 노드가 지속적으로 이미지 데이터를 발행.
  - `/odom`: 위치 추정 노드가 로봇의 주행 기록(odometry) 정보를 계속해서 발행.
  - `/cmd_vel`: 제어 노드가 로봇의 목표 속도 명령을 발행.

### 2.2. 서비스 (Services)

**서비스(Service)**는 **요청(Request)과 응답(Response)**이 동기적으로 이루어지는 **일대일(one-to-one)** 통신 방식입니다. 서비스를 제공하는 서버(Server)와 서비스를 요청하는 클라이언트(Client)로 구성됩니다.

- **특징:**
  - **동기(Synchronous):** 클라이언트는 서버로부터 응답이 올 때까지 기다립니다. (Blocking call)
  - **일대일(One-to-One):** 특정 클라이언트의 요청은 하나의 서버에 의해 처리되고, 응답은 해당 클라이언트에게만 전달됩니다.
  - **서비스 데이터 타입:** 요청과 응답의 데이터 구조를 `.srv` 파일에 정의합니다.

- **사용 예시:**
  - `/set_camera_info`: 카메라의 설정을 변경해달라고 요청하고, 성공 여부를 응답받음.
  - `/clear_costmaps`: 내비게이션 시스템의 비용 지도를 초기화해달라고 요청.
  - `/spawn_entity`: 시뮬레이션 환경에 새로운 로봇이나 물체를 생성해달라고 요청.

### 2.3. 액션 (Actions)

**액션(Action)**은 **장시간 실행되는 목표(Goal)**를 비동기적으로 처리하고, **지속적인 피드백(Feedback)**과 **최종 결과(Result)**를 제공하는 통신 방식입니다. 액션을 제공하는 액션 서버(Action Server)와 목표를 요청하는 액션 클라이언트(Action Client)로 구성됩니다.

- **특징:**
  - **비동기(Asynchronous):** 클라이언트는 목표를 보낸 후 다른 작업을 계속할 수 있으며, 중간 피드백을 받을 수 있습니다.
  - **취소 가능(Preemptible):** 클라이언트는 진행 중인 목표를 언제든지 취소할 수 있습니다.
  - **피드백 제공:** 서버는 목표를 달성하는 동안 현재 진행 상태(예: 이동한 거리, 남은 시간)를 클라이언트에게 지속적으로 보낼 수 있습니다.
  - **액션 데이터 타입:** 목표, 결과, 피드백의 데이터 구조를 `.action` 파일에 정의합니다.

- **사용 예시:**
  - `/navigate_to_pose`: 내비게이션 시스템에 특정 좌표로 이동하라는 목표를 전달. 서버는 이동하는 동안 현재 위치를 피드백으로 보내고, 도착 시 성공 여부를 결과로 반환.
  - `/rotate_absolute`: 로봇 팔을 특정 각도로 회전시키라는 목표를 전달.
  - `/backup`: 로봇을 일정 거리만큼 후진시키는 목표를 전달.

## 3. 파라미터 (Parameters)

**파라미터(Parameter)**는 노드가 실행 중에 외부에서 설정하거나 변경할 수 있는 **설정값**입니다. 각 노드는 자신만의 파라미터를 가질 수 있으며, 이를 통해 코드 수정 없이 노드의 동작을 유연하게 변경할 수 있습니다.

- **특징:**
  - **동적 재설정(Dynamic Reconfigure):** 노드가 실행되는 동안에도 파라미터 값을 변경할 수 있습니다.
  - **타입 지원:** 문자열, 정수, 실수, 불리언 등 다양한 데이터 타입을 지원합니다.
  - **저장 및 로드:** 파라미터 값들을 YAML 파일에 저장하고, 노드 실행 시 불러올 수 있습니다.

- **사용 예시:**
  - 카메라 노드의 해상도(`width`, `height`) 설정
  - 제어 노드의 PID 게인(`kp`, `ki`, `kd`) 값 조절
  - 내비게이션 노드의 최대 속도(`max_velocity`) 제한

## 요약

| 통신 방식 | 주 용도 | 통신 형태 | 동기/비동기 | QoS 적용 |
| --- | --- | --- | --- | --- |
| **토픽** | 센서 데이터, 상태 정보 등 연속적인 데이터 스트림 | Publish/Subscribe | 비동기 | O |
| **서비스** | 특정 작업을 즉시 처리하고 결과를 받아야 할 때 | Request/Response | 동기 | X |
| **액션** | 시간이 오래 걸리는 작업을 실행하고 중간 피드백을 원할 때 | Goal/Feedback/Result | 비동기 | O |

이러한 핵심 개념들을 이해하는 것은 ROS 2 기반의 로봇 시스템을 설계하고 개발하는 데 있어 가장 중요한 첫걸음입니다.

---

## 4. DDS (Data Distribution Service)

ROS 2의 핵심 통신 기반구조는 **DDS(Data Distribution Service)**입니다. DDS는 실시간 분산 시스템을 위한 국제 표준 통신 미들웨어로, ROS 1의 중앙집중식 마스터 노드의 한계를 극복합니다.

### 4.1. DDS의 주요 특징

- **분산 구조**: 중앙 브로커 없이 피어-투-피어 통신
- **Quality of Service (QoS)**: 통신의 신뢰성과 성능을 세밀하게 제어
- **자동 발견**: 네트워크상의 다른 노드들을 자동으로 탐지
- **플랫폼 독립성**: 다양한 OS와 하드웨어에서 동작

### 4.2. QoS (Quality of Service) 정책

ROS 2에서는 통신의 품질을 세밀하게 제어할 수 있는 QoS 정책을 제공합니다.

#### 주요 QoS 정책들:

- **Reliability (신뢰성)**:
  - `RELIABLE`: 메시지 전달을 보장 (TCP와 유사)
  - `BEST_EFFORT`: 최선의 노력으로 전달, 손실 허용 (UDP와 유사)

- **Durability (지속성)**:
  - `TRANSIENT_LOCAL`: 늦게 가입한 구독자에게 최근 메시지 전달
  - `VOLATILE`: 실시간으로만 메시지 전달

- **History (히스토리)**:
  - `KEEP_LAST(n)`: 최근 n개 메시지만 유지
  - `KEEP_ALL`: 모든 메시지 유지

- **Deadline (데드라인)**:
  - 지정된 시간 내에 새 메시지가 도착해야 함을 명시

---

## 5. ROS 2 프로그래밍 실습

### 5.1. 간단한 Publisher 노드 (Python)

```python
import rclpy
from rclpy.node import Node
from std_msgs.msg import String

class MinimalPublisher(Node):
    def __init__(self):
        super().__init__('minimal_publisher')
        self.publisher_ = self.create_publisher(String, 'topic', 10)
        timer_period = 0.5  # seconds
        self.timer = self.create_timer(timer_period, self.timer_callback)
        self.i = 0

    def timer_callback(self):
        msg = String()
        msg.data = f'Hello World: {self.i}'
        self.publisher_.publish(msg)
        self.get_logger().info(f'Publishing: "{msg.data}"')
        self.i += 1

def main(args=None):
    rclpy.init(args=args)
    minimal_publisher = MinimalPublisher()
    rclpy.spin(minimal_publisher)
    minimal_publisher.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
```

### 5.2. 간단한 Subscriber 노드 (Python)

```python
import rclpy
from rclpy.node import Node
from std_msgs.msg import String

class MinimalSubscriber(Node):
    def __init__(self):
        super().__init__('minimal_subscriber')
        self.subscription = self.create_subscription(
            String,
            'topic',
            self.listener_callback,
            10)
        self.subscription  # prevent unused variable warning

    def listener_callback(self, msg):
        self.get_logger().info(f'I heard: "{msg.data}"')

def main(args=None):
    rclpy.init(args=args)
    minimal_subscriber = MinimalSubscriber()
    rclpy.spin(minimal_subscriber)
    minimal_subscriber.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
```

### 5.3. 서비스 서버 노드 (Python)

```python
from example_interfaces.srv import AddTwoInts
import rclpy
from rclpy.node import Node

class MinimalService(Node):
    def __init__(self):
        super().__init__('minimal_service')
        self.srv = self.create_service(
            AddTwoInts, 'add_two_ints', self.add_two_ints_callback)

    def add_two_ints_callback(self, request, response):
        response.sum = request.a + request.b
        self.get_logger().info(
            f'Incoming request\na: {request.a} b: {request.b}')
        return response

def main(args=None):
    rclpy.init(args=args)
    minimal_service = MinimalService()
    rclpy.spin(minimal_service)
    rclpy.shutdown()

if __name__ == '__main__':
    main()
```

### 5.4. 서비스 클라이언트 노드 (Python)

```python
from example_interfaces.srv import AddTwoInts
import rclpy
from rclpy.node import Node
import sys

class MinimalClientAsync(Node):
    def __init__(self):
        super().__init__('minimal_client_async')
        self.cli = self.create_client(AddTwoInts, 'add_two_ints')
        while not self.cli.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('service not available, waiting again...')
        self.req = AddTwoInts.Request()

    def send_request(self, a, b):
        self.req.a = a
        self.req.b = b
        return self.cli.call_async(self.req)

def main(args=None):
    rclpy.init(args=args)
    minimal_client = MinimalClientAsync()
    future = minimal_client.send_request(int(sys.argv[1]), int(sys.argv[2]))
    rclpy.spin_until_future_complete(minimal_client, future)
    response = future.result()
    minimal_client.get_logger().info(
        f'Result of add_two_ints: for {minimal_client.req.a} + {minimal_client.req.b} = {response.sum}')
    minimal_client.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
```

---

## 6. Launch 시스템

ROS 2의 **Launch 시스템**은 여러 노드를 동시에 시작하고 설정을 관리하는 도구입니다.

### 6.1. Launch 파일 예제 (Python)

```python
from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
        Node(
            package='demo_nodes_cpp',
            executable='talker',
            name='talker'),
        Node(
            package='demo_nodes_py',
            executable='listener',
            name='listener'),
    ])
```

### 6.2. 매개변수가 있는 Launch 파일

```python
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument(
            'node_prefix',
            default_value='my_robot',
            description='Prefix for node names'
        ),
        Node(
            package='my_package',
            executable='my_node',
            name=[LaunchConfiguration('node_prefix'), '_controller'],
            parameters=[{
                'use_sim_time': True,
                'max_velocity': 2.0
            }]
        )
    ])
```

---

## 7. 메시지와 인터페이스 정의

### 7.1. 커스텀 메시지 정의 (.msg)

```
# PersonInfo.msg
string name
int32 age
float64 height
bool is_student
string[] hobbies
geometry_msgs/Point position
```

### 7.2. 커스텀 서비스 정의 (.srv)

```
# CalculateArea.srv
# 요청 (Request)
float64 width
float64 height
---
# 응답 (Response)
float64 area
bool success
string message
```

### 7.3. 커스텀 액션 정의 (.action)

```
# MoveToPosition.action
# 목표 (Goal)
geometry_msgs/Point target_position
float64 max_velocity
---
# 결과 (Result)
geometry_msgs/Point final_position
float64 distance_traveled
bool success
---
# 피드백 (Feedback)
geometry_msgs/Point current_position
float64 distance_remaining
float64 elapsed_time
```

---

## 8. 실전 예제: 로봇 제어 시스템

### 8.1. 로봇 상태 모니터링 노드

```python
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import BatteryState, LaserScan
from geometry_msgs.msg import Twist
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy

class RobotMonitor(Node):
    def __init__(self):
        super().__init__('robot_monitor')
        
        # QoS 설정
        qos_profile = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            depth=10
        )
        
        # 센서 데이터 구독
        self.battery_subscription = self.create_subscription(
            BatteryState, '/battery_state', 
            self.battery_callback, qos_profile)
        
        self.laser_subscription = self.create_subscription(
            LaserScan, '/scan', 
            self.laser_callback, 10)
        
        # 제어 명령 발행
        self.cmd_publisher = self.create_publisher(
            Twist, '/cmd_vel', 10)
        
        # 타이머 설정
        self.timer = self.create_timer(0.1, self.control_callback)
        
        # 상태 변수
        self.battery_level = 100.0
        self.min_distance = float('inf')
        
    def battery_callback(self, msg):
        self.battery_level = msg.percentage * 100
        if self.battery_level < 20.0:
            self.get_logger().warn(f'Low battery: {self.battery_level:.1f}%')
    
    def laser_callback(self, msg):
        # 최소 거리 계산
        valid_ranges = [r for r in msg.ranges if msg.range_min < r < msg.range_max]
        if valid_ranges:
            self.min_distance = min(valid_ranges)
    
    def control_callback(self):
        cmd = Twist()
        
        # 배터리 부족 시 정지
        if self.battery_level < 15.0:
            cmd.linear.x = 0.0
            cmd.angular.z = 0.0
            self.get_logger().error('Critical battery level! Stopping robot.')
        
        # 장애물 감지 시 감속
        elif self.min_distance < 0.5:
            cmd.linear.x = 0.1
            cmd.angular.z = 0.5  # 회전하여 회피
            self.get_logger().info(f'Obstacle detected at {self.min_distance:.2f}m')
        
        # 정상 주행
        else:
            cmd.linear.x = 0.5
            cmd.angular.z = 0.0
        
        self.cmd_publisher.publish(cmd)

def main(args=None):
    rclpy.init(args=args)
    robot_monitor = RobotMonitor()
    rclpy.spin(robot_monitor)
    robot_monitor.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
```

---

## 예제 및 풀이 (Examples and Solutions)

### 예제 1: QoS 정책 선택 문제

**문제**: 다음 상황에서 적절한 QoS 정책을 선택하고 이유를 설명하시오.
- 상황 1: 로봇의 배터리 상태 정보 전송
- 상황 2: 카메라 이미지 스트림 전송  
- 상황 3: 로봇 정지 명령 전송

**풀이**:

**상황 1 - 배터리 상태**
```python
qos_battery = QoSProfile(
    reliability=ReliabilityPolicy.RELIABLE,
    durability=DurabilityPolicy.TRANSIENT_LOCAL,
    depth=1
)
```
- **Reliable**: 배터리 정보는 중요하므로 전달 보장
- **Transient Local**: 새로운 구독자도 최근 상태 확인 가능
- **Depth=1**: 최신 상태만 필요

**상황 2 - 카메라 이미지**
```python
qos_camera = QoSProfile(
    reliability=ReliabilityPolicy.BEST_EFFORT,
    durability=DurabilityPolicy.VOLATILE,
    depth=5
)
```
- **Best Effort**: 실시간성 중시, 일부 프레임 손실 허용
- **Volatile**: 과거 이미지는 불필요
- **Depth=5**: 네트워크 지연 대비 버퍼

**상황 3 - 정지 명령**
```python
qos_emergency = QoSProfile(
    reliability=ReliabilityPolicy.RELIABLE,
    durability=DurabilityPolicy.VOLATILE,
    depth=10
)
```
- **Reliable**: 안전을 위해 전달 보장 필수
- **Volatile**: 실시간 명령
- **Depth=10**: 중복 전송으로 확실성 확보

### 예제 2: 로봇 내비게이션 시스템 설계

**문제**: 자율 이동 로봇의 내비게이션 시스템을 위한 노드 구조와 통신 방식을 설계하시오.

**풀이**:

```python
# 시스템 아키텍처
class NavigationSystem:
    """
    노드 구조:
    1. sensor_fusion_node: 센서 데이터 융합
    2. localization_node: 위치 추정 (SLAM)
    3. path_planner_node: 전역 경로 계획
    4. local_planner_node: 지역 경로 계획
    5. controller_node: 모터 제어
    """
    pass

# 1. 센서 융합 노드
class SensorFusionNode(Node):
    def __init__(self):
        super().__init__('sensor_fusion')
        # 센서 데이터 구독
        self.lidar_sub = self.create_subscription(
            LaserScan, '/scan', self.lidar_callback, 10)
        self.odom_sub = self.create_subscription(
            Odometry, '/odom', self.odom_callback, 10)
        
        # 융합 데이터 발행
        self.fused_pub = self.create_publisher(
            PoseWithCovarianceStamped, '/robot_pose', 10)

# 2. 경로 계획 노드 (액션 서버)
class PathPlannerNode(Node):
    def __init__(self):
        super().__init__('path_planner')
        self._action_server = ActionServer(
            self, NavigateToPose, 'navigate_to_pose',
            self.execute_callback)
    
    def execute_callback(self, goal_handle):
        # A* 알고리즘 등을 이용한 경로 계획
        # 피드백: 계획 진행률
        # 결과: 최종 경로
        pass

# 3. 제어 노드
class ControllerNode(Node):
    def __init__(self):
        super().__init__('controller')
        # 경로 구독
        self.path_sub = self.create_subscription(
            Path, '/planned_path', self.path_callback, 10)
        # 제어 명령 발행
        self.cmd_pub = self.create_publisher(
            Twist, '/cmd_vel', 10)
        
        # PID 제어기 파라미터
        self.declare_parameter('kp_linear', 1.0)
        self.declare_parameter('kp_angular', 2.0)
```

**통신 방식 선택**:
- **토픽**: 센서 데이터, 상태 정보 (연속적 데이터)
- **서비스**: 맵 저장/로드 (일회성 요청)
- **액션**: 내비게이션 명령 (장시간 실행, 진행률 피드백)

### 예제 3: ROS 2 성능 최적화

**문제**: 고주파수 센서 데이터를 처리하는 ROS 2 시스템의 성능을 최적화하는 방법을 제시하시오.

**풀이**:

```python
# 1. Zero-Copy 통신 사용
from rclpy.qos import qos_profile_sensor_data

class HighFrequencyNode(Node):
    def __init__(self):
        super().__init__('high_frequency_node')
        
        # 센서 데이터 전용 QoS 프로파일 사용
        self.subscription = self.create_subscription(
            Image, '/camera/image_raw',
            self.image_callback,
            qos_profile_sensor_data  # Best effort, volatile
        )
        
        # 콜백 그룹으로 병렬 처리
        from rclpy.callback_groups import MutuallyExclusiveCallbackGroup
        self.callback_group = MutuallyExclusiveCallbackGroup()
        
        self.timer = self.create_timer(
            0.01,  # 100Hz
            self.timer_callback,
            callback_group=self.callback_group
        )
    
    def image_callback(self, msg):
        # 이미지 처리를 별도 스레드에서 실행
        import threading
        thread = threading.Thread(
            target=self.process_image, args=(msg,))
        thread.start()
    
    def process_image(self, image_msg):
        # CPU 집약적인 이미지 처리
        pass

# 2. 컴포지션 노드로 통신 오버헤드 감소
class ComposedNodes(Node):
    def __init__(self):
        super().__init__('composed_nodes')
        
        # 같은 프로세스 내에서 직접 함수 호출
        self.sensor_processor = SensorProcessor()
        self.data_filter = DataFilter()
        
        self.timer = self.create_timer(0.01, self.process_pipeline)
    
    def process_pipeline(self):
        # 네트워크 통신 없이 직접 데이터 전달
        raw_data = self.sensor_processor.get_data()
        filtered_data = self.data_filter.filter(raw_data)
        self.publish_result(filtered_data)

# 3. 실시간 우선순위 설정
import os

def main(args=None):
    rclpy.init(args=args)
    
    # 실시간 우선순위 설정 (Linux)
    if os.name == 'posix':
        import sched
        param = sched.sched_param(50)  # 높은 우선순위
        try:
            sched.sched_setscheduler(
                0, sched.SCHED_FIFO, param)
        except PermissionError:
            print("실시간 우선순위 설정 권한 없음")
    
    node = HighFrequencyNode()
    
    # 멀티스레드 실행자 사용
    from rclpy.executors import MultiThreadedExecutor
    executor = MultiThreadedExecutor(num_threads=4)
    executor.add_node(node)
    executor.spin()
    
    node.destroy_node()
    rclpy.shutdown()
```

**최적화 기법 요약**:
1. **QoS 최적화**: 용도에 맞는 신뢰성/지연시간 설정
2. **Zero-Copy**: 대용량 데이터의 메모리 복사 최소화
3. **컴포지션**: 노드 간 통신 오버헤드 감소
4. **멀티스레딩**: 콜백 병렬 처리
5. **실시간 스케줄링**: OS 레벨 우선순위 조정
