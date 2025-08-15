# CNN의 핵심 구성요소

합성곱 신경망(Convolutional Neural Network, CNN)은 주로 이미지 및 비디오와 같은 그리드(grid) 형태의 데이터를 처리하는 데 특화된 딥러닝 모델입니다. CNN은 인간의 시신경 구조를 모방하여, 데이터의 공간적인 특징(spatial feature)을 효과적으로 추출하고 학습합니다.

CNN의 핵심 구성요소는 **합성곱 레이어(Convolutional Layer)**와 **풀링 레이어(Pooling Layer)**입니다.

---

## 1. 합성곱 레이어 (Convolutional Layer)

합성곱 레이어는 CNN의 가장 중요한 부분으로, 입력 데이터로부터 특징을 추출하는 역할을 합니다. 이는 **필터(Filter)** 또는 **커널(Kernel)**이라고 불리는 작은 행렬을 사용하여 입력 데이터를 순회하며 **합성곱(Convolution)** 연산을 수행함으로써 이루어집니다.

### 1.1. 수학적 정의

**2D 합성곱 연산:**
입력 이미지 $\mathbf{I}$와 커널 $\mathbf{K}$에 대한 합성곱 연산은 다음과 같이 정의됩니다:

$$(\mathbf{I} * \mathbf{K})(i,j) = \sum_{m}\sum_{n} \mathbf{I}(i+m, j+n) \cdot \mathbf{K}(m,n)$$

또는 상관관계(Cross-correlation, 실제 딥러닝에서 사용):
$$(\mathbf{I} \star \mathbf{K})(i,j) = \sum_{m}\sum_{n} \mathbf{I}(i-m, j-n) \cdot \mathbf{K}(m,n)$$

**다중 채널 합성곱:**
입력이 $C$ 채널을 가지는 경우:
$$\mathbf{Y}(i,j) = \sum_{c=1}^{C} (\mathbf{I}_c \star \mathbf{K}_c)(i,j) + b$$

여기서 $b$는 편향(bias) 항입니다.

### 1.2. 주요 개념

**필터 (Filter) / 커널 (Kernel):**
- 이미지의 특정 특징(예: 수직선, 수평선, 특정 색상, 질감 등)을 감지하기 위한 작은 크기의 학습 가능한 가중치 행렬
- **일반적인 크기**: 3×3, 5×5, 7×7 (홀수 크기 선호)
- **깊이**: 입력 채널 수와 동일
- **개수**: 출력 채널 수 결정

**특징적 필터 예시:**
```python
# 수직 엣지 탐지 필터
vertical_edge = np.array([[-1, 0, 1],
                         [-1, 0, 1], 
                         [-1, 0, 1]])

# 수평 엣지 탐지 필터  
horizontal_edge = np.array([[-1, -1, -1],
                           [ 0,  0,  0],
                           [ 1,  1,  1]])

# 가우시안 블러 필터
gaussian_blur = np.array([[1, 2, 1],
                         [2, 4, 2],
                         [1, 2, 1]]) / 16
```

**특징 맵 (Feature Map):**
- 하나의 필터가 입력 데이터 전체를 순회하며 합성곱 연산을 수행한 결과
- 특정 특징이 이미지의 어느 위치에서 활성화되는지를 나타내는 맵
- **차원**: (Height, Width, Channels)

**스트라이드 (Stride):**
- 필터가 입력을 순회할 때 이동하는 간격
- **일반적 값**: 1, 2, 3
- **효과**: 출력 크기 감소, 계산량 감소, 다운샘플링

**패딩 (Padding):**
- 입력의 가장자리에 값을 추가하는 기법
- **Zero Padding**: 0으로 채우기 (가장 일반적)
- **Same Padding**: 출력 크기를 입력과 동일하게 유지
- **Valid Padding**: 패딩 없음

**출력 크기 계산:**
$$\text{Output Size} = \frac{\text{Input Size} - \text{Kernel Size} + 2 \times \text{Padding}}{\text{Stride}} + 1$$

예시:
- 입력: 32×32, 커널: 5×5, 패딩: 2, 스트라이드: 1
- 출력: $\frac{32 - 5 + 2 \times 2}{1} + 1 = 32$

### 1.3. 파라미터와 계산량

**파라미터 수:**
$$\text{Parameters} = (\text{Kernel Height} \times \text{Kernel Width} \times \text{Input Channels} + 1) \times \text{Output Channels}$$

**계산량 (FLOPs):**
$$\text{FLOPs} = \text{Output Height} \times \text{Output Width} \times \text{Kernel Height} \times \text{Kernel Width} \times \text{Input Channels} \times \text{Output Channels}$$

### 1.4. 합성곱의 장점

1. **지역적 연결성 (Local Connectivity)**: 각 뉴런은 입력의 작은 영역에만 연결
2. **가중치 공유 (Weight Sharing)**: 동일한 필터가 전체 입력에 적용
3. **평행 이동 불변성 (Translation Invariance)**: 객체 위치 변화에 강건함
4. **파라미터 효율성**: 완전 연결 레이어 대비 적은 파라미터

---

## 2. 풀링 레이어 (Pooling Layer)

풀링 레이어는 합성곱 레이어에서 추출된 특징 맵의 크기를 줄여(Downsampling), 계산량을 감소시키고 모델의 일반화 성능을 높이는 역할을 합니다.

### 2.1. 수학적 정의

**최대 풀링 (Max Pooling):**
$$\text{MaxPool}(\mathbf{X})_{i,j} = \max_{m,n \in \text{Pool Region}} \mathbf{X}_{i \cdot s + m, j \cdot s + n}$$

**평균 풀링 (Average Pooling):**
$$\text{AvgPool}(\mathbf{X})_{i,j} = \frac{1}{k \times k} \sum_{m=0}^{k-1} \sum_{n=0}^{k-1} \mathbf{X}_{i \cdot s + m, j \cdot s + n}$$

여기서 $k$는 풀링 커널 크기, $s$는 스트라이드입니다.

### 2.2. 주요 목적과 효과

1. **차원 축소 (Dimensionality Reduction):**
   - 특징 맵의 공간적 크기를 줄여 계산량과 메모리 사용량 감소
   - 후속 레이어의 파라미터 수 감소

2. **평행 이동 불변성 (Translation Invariance):**
   - 입력의 작은 위치 변화에 강건한 특성 부여
   - 객체의 정확한 위치보다는 존재 여부에 집중

3. **과적합 방지 (Overfitting Prevention):**
   - 불필요한 세부 정보 제거로 일반화 성능 향상
   - 노이즈에 대한 강건성 증가

4. **수용 영역 증가 (Receptive Field Expansion):**
   - 상위 레이어가 더 넓은 영역의 정보를 볼 수 있게 함

### 2.3. 풀링 방법별 특성

**최대 풀링 (Max Pooling):**
- **특징:** 특정 영역에서 가장 강한 활성화 값만 선택
- **장점:**
  - 가장 두드러진 특징 보존
  - 노이즈에 강건
  - 희소성(Sparsity) 유지
- **단점:**
  - 정보 손실 가능성
  - 위치 정보 완전 손실
- **사용 사례:** 이미지 분류, 객체 탐지

**평균 풀링 (Average Pooling):**
- **특징:** 영역 내 모든 값의 평균 계산
- **장점:**
  - 부드러운 특징 추출
  - 모든 정보 활용
  - 안정적인 출력
- **단점:**
  - 중요한 특징이 희석될 가능성
  - 약한 신호 손실
- **사용 사례:** 전역 평균 풀링(GAP), 최종 분류층

**전역 평균 풀링 (Global Average Pooling, GAP):**
- **정의:** 전체 특징 맵을 하나의 값으로 축약
- **수식:** $\text{GAP}(\mathbf{X}) = \frac{1}{H \times W} \sum_{i=1}^{H} \sum_{j=1}^{W} \mathbf{X}_{i,j}$
- **장점:**
  - 완전연결층 대체로 파라미터 수 대폭 감소
  - 과적합 방지
  - 입력 크기에 무관
- **사용:** ResNet, Inception 등 현대 아키텍처

### 2.4. 출력 크기 계산

풀링 레이어의 출력 크기:
$$\text{Output Size} = \frac{\text{Input Size} - \text{Pool Size}}{\text{Stride}} + 1$$

예시:
- 입력: 32×32, 풀링: 2×2, 스트라이드: 2
- 출력: $\frac{32 - 2}{2} + 1 = 16$

### 2.5. PyTorch 구현 예제

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

# 최대 풀링 레이어
max_pool = nn.MaxPool2d(kernel_size=2, stride=2)

# 평균 풀링 레이어
avg_pool = nn.AvgPool2d(kernel_size=2, stride=2)

# 적응적 풀링 (출력 크기 고정)
adaptive_max_pool = nn.AdaptiveMaxPool2d((7, 7))
adaptive_avg_pool = nn.AdaptiveAvgPool2d((1, 1))  # GAP

# 사용 예제
x = torch.randn(1, 64, 32, 32)  # (배치, 채널, 높이, 너비)

# 기본 풀링
max_out = max_pool(x)  # (1, 64, 16, 16)
avg_out = avg_pool(x)  # (1, 64, 16, 16)

# 적응적 풀링
adaptive_max_out = adaptive_max_pool(x)  # (1, 64, 7, 7)
gap_out = adaptive_avg_pool(x)  # (1, 64, 1, 1)

# 함수형 API 사용
max_out_f = F.max_pool2d(x, kernel_size=2, stride=2)
avg_out_f = F.avg_pool2d(x, kernel_size=2, stride=2)
```

### 2.6. 최신 대안들

**Strided Convolution:**
- 풀링 대신 스트라이드 > 1인 합성곱 사용
- 학습 가능한 다운샘플링
- 더 나은 특징 보존 가능

**Dilated/Atrous Convolution:**
- 풀링 없이 수용 영역 확대
- 해상도 유지하면서 문맥 정보 확보
- 세밀한 분할 작업에 유용

**Attention Mechanism:**
- 중요한 영역에 가중치 부여
- 위치별 중요도 학습
- 전역적 문맥 고려

---

## 3. 활성화 함수 (Activation Functions)

활성화 함수는 CNN의 각 레이어에서 비선형성을 도입하여 복잡한 패턴을 학습할 수 있게 하는 핵심 구성요소입니다.

### 3.1. ReLU (Rectified Linear Unit)
가장 널리 사용되는 활성화 함수로, CNN의 표준이 되었습니다.

**수식:**
$$\text{ReLU}(x) = \max(0, x) = \begin{cases} x & \text{if } x \geq 0 \\ 0 & \text{if } x < 0 \end{cases}$$

**장점:**
- 계산이 매우 간단하고 빠름
- 기울기 소실 문제 완화
- 희소성(Sparsity) 제공으로 효율적 표현

**단점:**
- Dying ReLU 문제: 음수 입력에서 기울기가 0이 되어 뉴런이 죽을 수 있음

### 3.2. Leaky ReLU
ReLU의 Dying 문제를 해결한 변형입니다.

**수식:**
$$\text{LeakyReLU}(x) = \begin{cases} x & \text{if } x \geq 0 \\ \alpha x & \text{if } x < 0 \end{cases}$$

여기서 $\alpha$는 작은 상수 (보통 0.01)입니다.

### 3.3. ELU (Exponential Linear Unit)
**수식:**
$$\text{ELU}(x) = \begin{cases} x & \text{if } x \geq 0 \\ \alpha(e^x - 1) & \text{if } x < 0 \end{cases}$$

**특징:**
- 음수 영역에서 부드러운 곡선
- 평균이 0에 가까운 출력
- 더 강한 정규화 효과

### 3.4. Swish/SiLU (Sigmoid Linear Unit)
**수식:**
$$\text{Swish}(x) = x \cdot \sigma(x) = x \cdot \frac{1}{1 + e^{-x}}$$

**특징:**
- 자기 게이팅(Self-gating) 특성
- 부드러운 곡선으로 더 나은 기울기 흐름
- 최신 모델(EfficientNet 등)에서 사용

### 3.5. PyTorch 구현 예제

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

# 다양한 활성화 함수들
relu = nn.ReLU()
leaky_relu = nn.LeakyReLU(negative_slope=0.01)
elu = nn.ELU(alpha=1.0)
silu = nn.SiLU()  # Swish

# 함수형 API 사용
x = torch.randn(1, 64, 32, 32)
relu_out = F.relu(x)
leaky_relu_out = F.leaky_relu(x, negative_slope=0.01)
elu_out = F.elu(x, alpha=1.0)
silu_out = F.silu(x)
```

---

## 4. 배치 정규화 (Batch Normalization)

배치 정규화는 CNN 훈련을 안정화하고 가속화하는 중요한 기법입니다.

### 4.1. 수학적 정의

미니배치 $\mathcal{B} = \{x_1, x_2, \ldots, x_m\}$에 대해:

**평균과 분산 계산:**
$$\mu_{\mathcal{B}} = \frac{1}{m}\sum_{i=1}^{m} x_i$$
$$\sigma_{\mathcal{B}}^2 = \frac{1}{m}\sum_{i=1}^{m} (x_i - \mu_{\mathcal{B}})^2$$

**정규화:**
$$\hat{x}_i = \frac{x_i - \mu_{\mathcal{B}}}{\sqrt{\sigma_{\mathcal{B}}^2 + \epsilon}}$$

**스케일과 시프트:**
$$y_i = \gamma \hat{x}_i + \beta$$

여기서 $\gamma$와 $\beta$는 학습 가능한 파라미터입니다.

### 4.2. 주요 효과

1. **내부 공변량 시프트(Internal Covariate Shift) 감소:**
   - 각 레이어 입력의 분포를 안정화
   - 더 안정적인 기울기 흐름

2. **학습 속도 향상:**
   - 더 높은 학습률 사용 가능
   - 더 빠른 수렴

3. **정규화 효과:**
   - 드롭아웃의 필요성 감소
   - 과적합 방지

4. **가중치 초기화에 덜 민감:**
   - 초기화 방법에 대한 의존성 감소

### 4.3. CNN에서의 배치 정규화

2D 합성곱에서는 채널별로 정규화를 수행합니다:
- 입력 형태: $(N, C, H, W)$
- 각 채널 $c$에 대해 $(N, H, W)$ 차원에서 평균과 분산 계산

### 4.4. 배치 정규화 위치

**일반적인 순서:**
```
Conv → BatchNorm → Activation → Pool
```

**ResNet 스타일 (Pre-activation):**
```
BatchNorm → Activation → Conv
```

### 4.5. PyTorch 구현 예제

```python
import torch
import torch.nn as nn

class ConvBNReLU(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super(ConvBNReLU, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x

# 사용 예제
block = ConvBNReLU(3, 64, kernel_size=3, padding=1)
x = torch.randn(32, 3, 224, 224)  # (배치, 채널, 높이, 너비)
output = block(x)  # (32, 64, 224, 224)
```

### 4.6. 추론 시 동작

훈련 시와 달리 추론 시에는 이동 평균을 사용합니다:
- $\mu_{running}$: 훈련 중 계산된 평균의 이동 평균
- $\sigma_{running}^2$: 훈련 중 계산된 분산의 이동 평균

**추론 시 정규화:**
$$y = \gamma \frac{x - \mu_{running}}{\sqrt{\sigma_{running}^2 + \epsilon}} + \beta$$

---

## 5. 일반적인 CNN 구조

### 5.1. 기본 CNN 아키텍처 패턴

```
Input Image
    ↓
[Conv → BatchNorm → ReLU → Pool] × N
    ↓
[Conv → BatchNorm → ReLU → Pool] × M
    ↓
Global Average Pooling / Flatten
    ↓
Fully Connected Layer(s)
    ↓
Output (Classification/Regression)
```

### 5.2. 현대적 CNN 블록 구조

```python
class ModernCNNBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ModernCNNBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.pool = nn.MaxPool2d(2, 2)
        
    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.pool(x)
        return x
```

### 5.3. 설계 원칙

1. **점진적 특징 추상화:**
   - 얕은 층: 저수준 특징 (엣지, 텍스처)
   - 깊은 층: 고수준 특징 (객체 부분, 전체 객체)

2. **공간 해상도와 채널 수의 트레이드오프:**
   - 네트워크가 깊어질수록 공간 크기 ↓, 채널 수 ↑
   - 계산량과 표현력의 균형

3. **정보 병목 방지:**
   - 급격한 차원 축소 지양
   - 점진적 크기 감소

---

## 예제 및 풀이 (Examples and Solutions)

### 예제 1: 합성곱 및 풀링 연산

**문제:** 다음과 같은 4x4 입력 데이터와 3x3 필터가 주어졌을 때, Stride=1, Padding=0 조건에서의 합성곱 연산 결과(특징 맵)를 계산하시오. 이어서, 계산된 특징 맵에 2x2 Max Pooling(Stride=2)을 적용한 최종 결과를 구하시오.

- **입력 데이터 (4x4):**
  ```
  [[1, 2, 3, 0],
   [0, 1, 2, 3],
   [3, 0, 1, 2],
   [2, 3, 0, 1]]
  ```
- **필터 (3x3):**
  ```
  [[1, 0, 1],
   [0, 1, 0],
   [1, 0, 1]]
  ```

**풀이:**

**1. 합성곱 연산:**
필터를 입력 데이터의 좌측 상단부터 한 칸씩(Stride=1) 이동하며 요소별 곱셈(element-wise product)의 합을 계산합니다. 출력 특징 맵의 크기는 `(4-3+1) x (4-3+1) = 2x2`가 됩니다.

- **위치 (0,0):**
  - `(1*1)+(2*0)+(3*1) + (0*0)+(1*1)+(2*0) + (3*1)+(0*0)+(1*1) = 1+3+1+3+1 = 9`
- **위치 (0,1):**
  - `(2*1)+(3*0)+(0*1) + (1*0)+(2*1)+(3*0) + (0*1)+(1*0)+(2*1) = 2+2+2 = 6`
- **위치 (1,0):**
  - `(0*1)+(1*0)+(2*1) + (3*0)+(0*1)+(1*0) + (2*1)+(3*0)+(0*1) = 2+2 = 4`
- **위치 (1,1):**
  - `(1*1)+(2*0)+(3*1) + (0*0)+(1*1)+(2*0) + (3*1)+(0*0)+(1*1) = 1+3+1+3+1 = 9`

- **합성곱 결과 (특징 맵, 2x2):**
  ```
  [[9, 6],
   [4, 9]]
  ```

**2. 최대 풀링 연산:**
계산된 2x2 특징 맵에 2x2 Max Pooling(Stride=2)을 적용합니다. 이 경우 필터가 겹치지 않고 한 번만 적용됩니다.

- 2x2 영역 `[[9, 6], [4, 9]]`에서 가장 큰 값은 **9**입니다.

**답:**
- **합성곱 결과:** `[[9, 6], [4, 9]]`
- **최대 풀링 결과:** `[[9]]`
