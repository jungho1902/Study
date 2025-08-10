# ROS 2 개발 환경 및 도구

ROS 2로 로봇 애플리케이션을 효율적으로 개발하고 디버깅하기 위해서는 다양한 개발 도구의 사용법을 익히는 것이 중요합니다. 이 문서는 ROS 2의 표준 빌드 시스템인 `colcon`, 시각화 도구 `Rviz2`, 그리고 데이터 로깅 및 재생 도구인 `ros2 bag`에 대해 설명합니다.

## 1. Colcon 빌드 시스템 (Colcon Build System)

**Colcon**은 "collective construction"의 약자로, ROS 2의 표준 빌드 시스템입니다. 여러 개의 패키지를 한 번에 빌드하고 테스트할 수 있는 유연하고 확장 가능한 도구입니다.

- **워크스페이스 (Workspace):** ROS 2 개발은 워크스페이스라는 특정 디렉토리 구조 안에서 이루어집니다.
  ```
  workspace_folder/
  └── src/
      ├── package_1/
      │   ├── package.xml
      │   ├── CMakeLists.txt (C++) / setup.py (Python)
      │   └── ...
      └── package_2/
          ├── package.xml
          └── ...
  ```
  - `src`: 소스 코드를 위치시키는 디렉토리.
  - `build`: 빌드 과정에서 생성되는 중간 파일들이 저장되는 디렉토리.
  - `install`: 빌드가 완료된 후 실행 파일, 라이브러리 등이 설치되는 디렉토리.
  - `log`: 빌드 로그가 저장되는 디렉토리.

- **주요 명령어:**
  - **`colcon build`**: 워크스페이스(`src` 폴더) 내의 모든 패키지를 빌드합니다.
    ```bash
    # 워크스페이스 최상위 디렉토리에서 실행
    colcon build
    ```
  - **`colcon build --packages-select <package_name>`**: 특정 패키지만 선택하여 빌드합니다.
    ```bash
    colcon build --packages-select my_robot_controller
    ```
  - **`source install/setup.bash`**: 빌드된 패키지를 현재 터미널 세션에서 사용 가능하도록 환경을 설정합니다. 이 과정(sourcing)은 새 터미널을 열 때마다 필요합니다.
    ```bash
    source install/setup.bash
    ```

## 2. Rviz2 시각화 도구 (Rviz2 Visualization Tool)

**Rviz2**는 3D 로봇 모델, 센서 데이터, 좌표계, 계획 경로 등 다양한 로봇 데이터를 시각화하는 강력한 도구입니다. 복잡한 로봇 시스템의 상태를 직관적으로 파악하고 디버깅하는 데 필수적입니다.

- **실행 방법:**
  ```bash
  # 새 터미널에서 실행
  rviz2
  ```

- **주요 기능:**
  - **디스플레이 플러그인 (Display Plugins):** 왼쪽 'Displays' 패널에서 'Add' 버튼을 눌러 시각화할 데이터의 종류를 추가할 수 있습니다.
    - **RobotModel:** URDF 파일로 정의된 로봇의 3D 모델을 표시합니다.
    - **LaserScan:** LiDAR 센서 데이터를 2D/3D 점으로 표시합니다.
    - **Image:** 카메라 이미지를 표시합니다.
    - **TF:** 로봇의 모든 좌표계(frame)와 그 관계를 시각적으로 보여줍니다.
    - **Map:** SLAM이나 내비게이션으로 생성된 지도를 표시합니다.
    - **Path:** 로봇의 계획된 경로 또는 지나온 경로를 선으로 표시합니다.
  - **토픽 구독:** 각 디스플레이 플러그인은 특정 토픽의 메시지를 구독하여 데이터를 시각화합니다. 예를 들어, LaserScan 디스플레이는 `/scan` 토픽을 구독합니다.
  - **설정 저장/로드:** 현재 Rviz2의 디스플레이 구성과 설정을 `.rviz` 파일로 저장하고 나중에 다시 불러올 수 있어 편리합니다.

## 3. ros2 bag 데이터 로깅 (ros2 bag Data Logging)

**`ros2 bag`**은 ROS 2 시스템에서 오고 가는 메시지(토픽 데이터)를 파일에 기록(recording)하고, 나중에 그대로 재생(playing)할 수 있게 해주는 도구입니다.

- **주요 용도:**
  - **실험 데이터 수집:** 실제 로봇을 구동하며 얻은 센서 데이터를 기록하여 나중에 분석하거나 알고리즘 테스트에 사용.
  - **디버깅:** 특정 상황에서 발생하는 문제를 재현하기 위해 당시의 모든 토픽 데이터를 기록.
  - **알고리즘 개발:** 실제 로봇 없이 기록된 데이터만으로 새로운 알고리즘을 테스트.

- **주요 명령어:**
  - **`ros2 bag record -o <bag_name> <topic_1> <topic_2> ...`**: 특정 토픽들을 지정된 이름의 bag 파일에 기록합니다.
    ```bash
    # /scan, /odom 토픽의 데이터를 my_bag 폴더에 저장
    ros2 bag record -o my_bag /scan /odom
    ```
  - **`ros2 bag record -a`**: 현재 발행되고 있는 모든 토픽을 기록합니다.
    ```bash
    ros2 bag record -a
    ```
  - **`ros2 bag play <bag_name>`**: 기록된 bag 파일을 재생합니다. bag 파일에 저장된 메시지들이 원래의 시간 간격에 맞춰 그대로 다시 발행됩니다.
    ```bash
    ros2 bag play my_bag
    ```
  - **`ros2 bag info <bag_name>`**: bag 파일에 대한 정보(기록된 토픽, 메시지 수, 용량 등)를 보여줍니다.
    ```bash
    ros2 bag info my_bag
    ```

이 세 가지 도구(`colcon`, `Rviz2`, `ros2 bag`)는 ROS 2 개발의 핵심이며, 능숙하게 사용하는 것이 생산성을 높이는 지름길입니다. 이 외에도 `ros2 topic`, `ros2 node`, `ros2 service`, `ros2 action` 등 다양한 CLI(Command-Line Interface) 도구들이 디버깅과 시스템 분석에 유용하게 사용됩니다.
