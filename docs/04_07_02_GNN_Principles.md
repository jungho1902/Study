# GNN의 기본 원리: 메시지 전달

## 개요

**그래프 신경망(GNN)**의 핵심 아이디어는 그래프의 각 노드가 자신의 정보와 **이웃 노드들의 정보를 반복적으로 집계하여** 자신의 표현(representation)을 업데이트하는 것입니다. 이 과정을 통해 노드는 자신의 지역적인 이웃(local neighborhood)을 넘어, 그래프 전체의 구조적인 정보를 점차적으로 반영하게 됩니다.

이러한 핵심 원리를 **메시지 전달(Message Passing)** 또는 **이웃 집계(Neighborhood Aggregation)**라고 부릅니다.

**수학적 프레임워크:**
주어진 그래프 $G = (V, E)$에서 각 노드 $v \in V$의 초기 특징을 $\mathbf{x}_v \in \mathbb{R}^d$라 할 때, GNN은 다음과 같은 반복적 과정을 통해 노드 표현을 학습합니다:

$$\mathbf{h}_v^{(l+1)} = \text{UPDATE}^{(l)}\left(\mathbf{h}_v^{(l)}, \text{AGGREGATE}^{(l)}\left(\{\mathbf{m}_{u,v}^{(l)} : u \in \mathcal{N}(v)\}\right)\right)$$

여기서:
- $\mathbf{h}_v^{(l)}$: l번째 레이어에서 노드 v의 표현 벡터
- $\mathcal{N}(v)$: 노드 v의 이웃 집합
- $\mathbf{m}_{u,v}^{(l)}$: 노드 u에서 v로 전달되는 메시지
- UPDATE, AGGREGATE: 학습 가능한 함수

## 1. 메시지 전달 과정 (Message Passing Framework)

메시지 전달 패러다임은 크게 세 단계로 구성되며, 이 과정은 GNN의 각 레이어에서 반복적으로 수행됩니다.

### 1.1. 메시지 생성 (Message Generation)

각 노드는 자신의 이웃 노드들에게 전달할 **메시지**를 생성합니다.

**기본 메시지 함수:**
$$\mathbf{m}_{u,v}^{(l)} = \text{MESSAGE}^{(l)}(\mathbf{h}_u^{(l)}, \mathbf{h}_v^{(l)}, \mathbf{e}_{u,v})$$

여기서:
- $\mathbf{h}_u^{(l)}$: 송신자 노드 u의 표현
- $\mathbf{h}_v^{(l)}$: 수신자 노드 v의 표현
- $\mathbf{e}_{u,v}$: 엣지 특징 (있는 경우)

**메시지 생성 방법들:**

**1. 단순 복사 (Simple Copy):**
$$\mathbf{m}_{u,v}^{(l)} = \mathbf{h}_u^{(l)}$$

**2. 선형 변환 (Linear Transformation):**
$$\mathbf{m}_{u,v}^{(l)} = \mathbf{W}^{(l)} \mathbf{h}_u^{(l)}$$

**3. 엣지 특징 포함 (Edge-Aware):**
$$\mathbf{m}_{u,v}^{(l)} = \text{MLP}([\mathbf{h}_u^{(l)}; \mathbf{h}_v^{(l)}; \mathbf{e}_{u,v}])$$

**4. 어텐션 기반 (Attention-based):**
$$\mathbf{m}_{u,v}^{(l)} = \alpha_{u,v}^{(l)} \mathbf{W}^{(l)} \mathbf{h}_u^{(l)}$$

### 1.2. 메시지 집계 (Message Aggregation)

각 노드는 이웃들로부터 받은 모든 메시지를 하나의 벡터로 **집계**합니다.

**집계 함수의 요구사항:**
- **순열 불변성 (Permutation Invariance):** 이웃 순서와 무관
- **크기 불변성 (Size Invariance):** 이웃 수와 무관한 결과

**주요 집계 함수들:**

**1. 합 집계 (Sum Aggregation):**
$$\mathbf{a}_v^{(l)} = \sum_{u \in \mathcal{N}(v)} \mathbf{m}_{u,v}^{(l)}$$

**2. 평균 집계 (Mean Aggregation):**
$$\mathbf{a}_v^{(l)} = \frac{1}{|\mathcal{N}(v)|} \sum_{u \in \mathcal{N}(v)} \mathbf{m}_{u,v}^{(l)}$$

**3. 최대값 집계 (Max Aggregation):**
$$\mathbf{a}_v^{(l)} = \max_{u \in \mathcal{N}(v)} \mathbf{m}_{u,v}^{(l)}$$

**4. LSTM 집계 (LSTM Aggregation):**
$$\mathbf{a}_v^{(l)} = \text{LSTM}(\{\mathbf{m}_{u,v}^{(l)} : u \in \mathcal{N}(v)\})$$

**5. 어텐션 집계 (Attention Aggregation):**
$$\mathbf{a}_v^{(l)} = \sum_{u \in \mathcal{N}(v)} \alpha_{u,v}^{(l)} \mathbf{m}_{u,v}^{(l)}$$

### 1.3. 노드 표현 업데이트 (Node Representation Update)

집계된 메시지와 현재 노드 표현을 결합하여 새로운 표현을 생성합니다.

**업데이트 함수:**
$$\mathbf{h}_v^{(l+1)} = \text{UPDATE}^{(l)}(\mathbf{h}_v^{(l)}, \mathbf{a}_v^{(l)})$$

**업데이트 방법들:**

**1. 연결 기반 (Concatenation-based):**
$$\mathbf{h}_v^{(l+1)} = \sigma(\mathbf{W}^{(l)} [\mathbf{h}_v^{(l)}; \mathbf{a}_v^{(l)}] + \mathbf{b}^{(l)})$$

**2. 잔차 연결 (Residual Connection):**
$$\mathbf{h}_v^{(l+1)} = \mathbf{h}_v^{(l)} + \sigma(\mathbf{W}^{(l)} \mathbf{a}_v^{(l)} + \mathbf{b}^{(l)})$$

**3. 게이트 기반 (Gate-based):**
$$\mathbf{z}_v = \sigma(\mathbf{W}_z [\mathbf{h}_v^{(l)}; \mathbf{a}_v^{(l)}])$$
$$\mathbf{h}_v^{(l+1)} = (1-\mathbf{z}_v) \odot \mathbf{h}_v^{(l)} + \mathbf{z}_v \odot \tilde{\mathbf{h}}_v^{(l+1)}$$

**4. LSTM 업데이트:**
$$\mathbf{h}_v^{(l+1)} = \text{LSTM}(\mathbf{a}_v^{(l)}, \mathbf{h}_v^{(l)})$$

## 2. GNN 레이어의 역할과 수용 영역 (Receptive Field)

### 2.1. 계층적 정보 집계

GNN의 핵심은 레이어를 통해 점진적으로 더 넓은 이웃 정보를 집계하는 것입니다.

**수용 영역의 확장:**
- **1개 레이어:** 각 노드는 1-hop 이웃들의 정보를 집계
  - $\mathbf{h}_v^{(1)} = f(\mathbf{h}_v^{(0)}, \{\mathbf{h}_u^{(0)} : u \in \mathcal{N}(v)\})$

- **2개 레이어:** 2-hop 이웃까지의 정보를 간접적으로 반영
  - $\mathbf{h}_v^{(2)} = f(\mathbf{h}_v^{(1)}, \{\mathbf{h}_u^{(1)} : u \in \mathcal{N}(v)\})$

- **K개 레이어:** K-hop 이웃까지의 정보 집계
  - 수용 영역: $\mathcal{R}_K(v) = \{u \in V : d(u,v) \leq K\}$

### 2.2. 정보 전파의 수학적 이해

**그래프 라플라시안과의 관계:**
GNN의 메시지 전달은 그래프 라플라시안의 거듭제곱과 유사한 정보 확산 패턴을 보입니다.

**스펙트럴 관점:**
- 인접 행렬 $\mathbf{A}$의 k제곱: $(\mathbf{A}^k)_{ij}$는 노드 i에서 j까지의 길이 k인 경로 수
- GNN의 k번째 레이어: k-hop 이웃의 정보를 집계

### 2.3. 과적합화 (Over-smoothing) 문제

**문제점:**
레이어가 깊어질수록 모든 노드의 표현이 비슷해지는 현상이 발생합니다.

**수학적 분석:**
$$\lim_{k \to \infty} \mathbf{H}^{(k)} = \mathbf{1}\mathbf{1}^T \mathbf{H}^{(0)} / n$$

**원인:**
- 정보의 과도한 평활화
- 기울기 소실 문제
- 노드별 고유 정보 손실

**해결책:**
- 잔차 연결 (Residual Connections)
- 드롭아웃 (Dropout)
- 배치 정규화 (Batch Normalization)
- 적응적 깊이 제어

## 3. 학습 목표와 다운스트림 태스크

### 3.1. 그래프 머신러닝 태스크 분류

GNN으로 학습된 노드 표현은 다양한 그래프 머신러닝 태스크에 활용됩니다.

#### 노드 수준 태스크 (Node-level Tasks)

**1. 노드 분류 (Node Classification):**
- 목표: 각 노드의 레이블 예측
- 손실 함수: $\mathcal{L} = \sum_{v \in V_{\text{labeled}}} \ell(y_v, f(\mathbf{h}_v))$
- 예시: 
  - 소셜 네트워크에서 사용자 관심사 분류
  - 논문 인용 네트워크에서 연구 분야 분류
  - 단백질 네트워크에서 기능 예측

**2. 노드 회귀 (Node Regression):**
- 목표: 각 노드의 연속값 예측
- 손실 함수: $\mathcal{L} = \sum_{v \in V} ||y_v - f(\mathbf{h}_v)||^2$
- 예시: 분자에서 원자별 특성 예측

#### 엣지 수준 태스크 (Edge-level Tasks)

**3. 링크 예측 (Link Prediction):**
- 목표: 노드 쌍 간 엣지 존재 확률 예측
- 예측 함수: $p(e_{uv}) = \sigma(\mathbf{h}_u^T \mathbf{h}_v)$ 또는 $p(e_{uv}) = \text{MLP}([\mathbf{h}_u; \mathbf{h}_v])$
- 손실 함수: $\mathcal{L} = -\sum_{(u,v)} y_{uv} \log p(e_{uv}) + (1-y_{uv}) \log(1-p(e_{uv}))$
- 예시:
  - 소셜 네트워크에서 친구 추천
  - 추천 시스템에서 사용자-아이템 상호작용 예측
  - 지식 그래프 완성

**4. 엣지 분류/회귀:**
- 목표: 엣지의 속성이나 타입 예측
- 예시: 소셜 네트워크에서 관계 유형 분류

#### 그래프 수준 태스크 (Graph-level Tasks)

**5. 그래프 분류 (Graph Classification):**
- 목표: 전체 그래프의 클래스 예측
- 그래프 표현: $\mathbf{h}_G = \text{READOUT}(\{\mathbf{h}_v : v \in V\})$
- READOUT 함수:
  - Sum: $\mathbf{h}_G = \sum_{v \in V} \mathbf{h}_v$
  - Mean: $\mathbf{h}_G = \frac{1}{|V|} \sum_{v \in V} \mathbf{h}_v$
  - Max: $\mathbf{h}_G = \max_{v \in V} \mathbf{h}_v$
  - Attention: $\mathbf{h}_G = \sum_{v \in V} \alpha_v \mathbf{h}_v$

**6. 그래프 회귀:**
- 목표: 그래프 속성의 연속값 예측
- 예시: 분자의 독성, 용해도, 결합 친화도 예측

### 3.2. 손실 함수와 최적화

**분류 태스크:**
$$\mathcal{L}_{\text{CE}} = -\sum_{i} \sum_{c} y_{i,c} \log(\hat{y}_{i,c})$$

**회귀 태스크:**
$$\mathcal{L}_{\text{MSE}} = \sum_{i} ||y_i - \hat{y}_i||^2$$

**링크 예측:**
$$\mathcal{L}_{\text{Link}} = -\sum_{(u,v) \in E} \log \sigma(\mathbf{h}_u^T \mathbf{h}_v) - \sum_{(u,v) \notin E} \log(1 - \sigma(\mathbf{h}_u^T \mathbf{h}_v))$$

### 3.3. 평가 지표

**분류:**
- 정확도 (Accuracy)
- F1-Score
- AUC-ROC
- Micro/Macro 평균

**회귀:**
- MSE, MAE
- R² Score
- RMSE

**링크 예측:**
- AUC
- Average Precision
- Hit@K

## 4. GNN의 이론적 기초

### 4.1. 표현력 (Expressive Power)

**Weisfeiler-Lehman (WL) 테스트와의 관계:**
- WL 테스트: 그래프 동형사상을 판별하는 알고리즘
- **정리:** 메시지 전달 GNN의 구별 능력은 1-WL 테스트와 동일함
- **한계:** 일부 그래프 구조(예: 삼각형, 사이클)를 구별하지 못함

**개선 방향:**
- 고차 GNN (Higher-order GNNs)
- 구조적 특징 추가 (Structural Features)
- 위치 인코딩 (Positional Encoding)

### 4.2. 일반화 능력

**귀납적 vs 변환적 학습:**
- **귀납적 (Inductive):** 새로운 노드/그래프에 일반화 가능
- **변환적 (Transductive):** 학습 시 보인 노드에만 적용

**Domain Adaptation:**
- 그래프 구조가 다른 도메인 간 전이 학습
- 적대적 학습을 통한 도메인 불변 특징 추출

## 5. 구현 고려사항

### 5.1. 계산 복잡도

**시간 복잡도:**
- 메시지 생성: $O(|E| \cdot d)$
- 집계: $O(|V| \cdot d \cdot \bar{d})$ (평균 차수 $\bar{d}$)
- 업데이트: $O(|V| \cdot d^2)$

**공간 복잡도:**
- 노드 표현: $O(|V| \cdot d)$
- 엣지 저장: $O(|E|)$

### 5.2. 배치 처리 (Batching)

**그래프 배치의 어려움:**
- 가변 크기
- 불규칙한 구조

**해결책:**
- 패딩 (Padding)
- 서브그래프 샘플링
- 그래프 패킹 (Graph Packing)

### 5.3. 확장성 (Scalability)

**대규모 그래프 처리:**
- 미니배치 학습
- 샘플링 기법 (FastGCN, GraphSampling)
- 분산 처리

GNN의 메시지 전달 메커니즘은 그래프 데이터의 구조적 정보를 효과적으로 학습하는 강력한 프레임워크입니다. 다음 섹션에서는 이 원리를 구체화한 다양한 GNN 아키텍처들을 살펴보겠습니다.
