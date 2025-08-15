# 주요 GNN 아키텍처

메시지 전달(Message Passing)이라는 기본 원리를 바탕으로, 이웃의 정보를 어떻게 집계(aggregate)하고 업데이트(update)할 것인지에 따라 다양한 그래프 신경망(GNN) 아키텍처가 제안되었습니다.

각 아키텍처는 다음과 같은 차별화된 설계 철학을 가집니다:
- **메시지 생성 방법**: 이웃 정보를 어떻게 인코딩할 것인가
- **집계 함수**: 여러 메시지를 어떻게 통합할 것인가  
- **업데이트 메커니즘**: 새로운 노드 표현을 어떻게 계산할 것인가
- **학습 패러다임**: Transductive vs Inductive 학습

---

## 1. GCN (Graph Convolutional Network)

### 1.1. 개념 및 배경
GCN은 **스펙트럴 그래프 이론(Spectral Graph Theory)**에서 출발하여 이미지 처리의 CNN 개념을 그래프에 적용한 첫 번째 성공적인 모델입니다.

### 1.2. 수학적 기초

**스펙트럴 관점:**
그래프 라플라시안 $L = D - A$의 고유값 분해:
$$L = U\Lambda U^T$$

**그래프 푸리에 변환:**
$$\hat{\mathbf{x}} = U^T \mathbf{x}$$

**스펙트럴 컨볼루션:**
$$g_\theta \star \mathbf{x} = U g_\theta(\Lambda) U^T \mathbf{x}$$

**첫 번째 근사 (Chebyshev Polynomial):**
$$g_\theta(\Lambda) \approx \sum_{k=0}^K \theta_k T_k(\tilde{\Lambda})$$

**최종 근사 (1차 근사):**
$$g_\theta \star \mathbf{x} \approx \theta_0 \mathbf{x} + \theta_1 (D^{-1/2}AD^{-1/2}) \mathbf{x}$$

### 1.3. GCN의 최종 형태

**레이어별 전파 규칙:**
$$\mathbf{H}^{(l+1)} = \sigma(\tilde{\mathbf{D}}^{-\frac{1}{2}}\tilde{\mathbf{A}}\tilde{\mathbf{D}}^{-\frac{1}{2}} \mathbf{H}^{(l)} \mathbf{W}^{(l)})$$

여기서:
- $\tilde{\mathbf{A}} = \mathbf{A} + \mathbf{I}$: 자기 루프가 추가된 인접 행렬
- $\tilde{\mathbf{D}}_{ii} = \sum_j \tilde{\mathbf{A}}_{ij}$: 차수 행렬
- $\mathbf{H}^{(l)} \in \mathbb{R}^{n \times d^{(l)}}$: l번째 레이어의 노드 특징 행렬
- $\mathbf{W}^{(l)} \in \mathbb{R}^{d^{(l)} \times d^{(l+1)}}$: 학습 가능한 가중치 행렬

### 1.4. 메시지 전달 해석

**메시지 생성:**
$$\mathbf{m}_{u,v}^{(l)} = \frac{1}{\sqrt{d_u d_v}} \mathbf{h}_u^{(l)}$$

**메시지 집계:**
$$\mathbf{a}_v^{(l)} = \sum_{u \in \mathcal{N}(v) \cup \{v\}} \mathbf{m}_{u,v}^{(l)}$$

**노드 업데이트:**
$$\mathbf{h}_v^{(l+1)} = \sigma(\mathbf{W}^{(l)} \mathbf{a}_v^{(l)})$$

### 1.5. 특성 및 한계

**장점:**
- 이론적 기반이 탄탄함 (스펙트럴 이론)
- 계산 효율성: $O(|E|)$ 시간 복잡도
- 간단한 구현

**단점:**
- **Transductive 학습**: 전체 그래프 구조 필요
- 고정된 그래프 구조에 의존
- 이웃 중요도 차별화 불가
- 깊은 네트워크에서 과평활화 문제

### 1.6. 구현 예시

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class GCNLayer(nn.Module):
    def __init__(self, in_features, out_features):
        super(GCNLayer, self).__init__()
        self.linear = nn.Linear(in_features, out_features)
        
    def forward(self, x, adj):
        # 정규화된 인접 행렬 계산
        degree = adj.sum(1)
        d_inv_sqrt = torch.pow(degree, -0.5)
        d_inv_sqrt[d_inv_sqrt == float('inf')] = 0
        d_mat_inv_sqrt = torch.diag(d_inv_sqrt)
        normalized_adj = d_mat_inv_sqrt @ adj @ d_mat_inv_sqrt
        
        # GCN 전파
        x = self.linear(x)
        x = normalized_adj @ x
        return F.relu(x)
```

---

## 2. GraphSAGE (Graph SAmple and aggreGatE)

### 2.1. 개념 및 동기
GraphSAGE는 GCN의 **Transductive 한계**를 극복하고, **Inductive Learning**과 **확장성(Scalability)**을 달성하기 위해 개발된 혁신적인 모델입니다.

### 2.2. 핵심 아이디어

**1. 이웃 샘플링 (Neighborhood Sampling):**
- 모든 이웃 대신 고정된 수 $S$개의 이웃을 샘플링
- 메모리 사용량과 계산량을 예측 가능하게 제어

**2. 집계 함수의 다양화:**
- 다양한 집계 함수로 이웃 정보를 통합
- 순열 불변성을 만족하는 함수 사용

### 2.3. 알고리즘

**GraphSAGE Forward Pass:**
```
for l = 1 to L:
    for v in V:
        # 1. 이웃 샘플링
        N_l(v) = SAMPLE(N(v), S_l)
        
        # 2. 집계
        h_{N(v)}^(l) = AGGREGATE_l({h_u^(l-1) : u in N_l(v)})
        
        # 3. 업데이트
        h_v^(l) = σ(W^(l) · CONCAT(h_v^(l-1), h_{N(v)}^(l)))
        
        # 4. 정규화 (선택적)
        h_v^(l) = h_v^(l) / ||h_v^(l)||_2
```

### 2.4. 집계 함수 (Aggregator Functions)

**1. Mean Aggregator:**
$$\mathbf{h}_{\mathcal{N}(v)}^{(l)} = \text{MEAN}(\{\mathbf{h}_u^{(l-1)} : u \in \mathcal{N}(v)\})$$

**2. LSTM Aggregator:**
$$\mathbf{h}_{\mathcal{N}(v)}^{(l)} = \text{LSTM}([\mathbf{h}_{u_1}^{(l-1)}, \mathbf{h}_{u_2}^{(l-1)}, ..., \mathbf{h}_{u_k}^{(l-1)}])$$

**3. Pooling Aggregator:**
$$\mathbf{h}_{\mathcal{N}(v)}^{(l)} = \max(\{\sigma(\mathbf{W}_{\text{pool}} \mathbf{h}_u^{(l-1)} + \mathbf{b}) : u \in \mathcal{N}(v)\})$$

**4. Self-Attention Aggregator:**
$$\alpha_{u,v} = \text{softmax}(\mathbf{a}^T [\mathbf{W}\mathbf{h}_u || \mathbf{W}\mathbf{h}_v])$$
$$\mathbf{h}_{\mathcal{N}(v)}^{(l)} = \sum_{u \in \mathcal{N}(v)} \alpha_{u,v} \mathbf{h}_u^{(l-1)}$$

### 2.5. 노드 업데이트

**연결 기반 업데이트:**
$$\mathbf{h}_v^{(l)} = \sigma(\mathbf{W}^{(l)} \cdot [\mathbf{h}_v^{(l-1)} || \mathbf{h}_{\mathcal{N}(v)}^{(l)}])$$

**L2 정규화:**
$$\mathbf{h}_v^{(l)} := \frac{\mathbf{h}_v^{(l)}}{||\mathbf{h}_v^{(l)}||_2}$$

### 2.6. 샘플링 전략

**균등 샘플링 (Uniform Sampling):**
- 모든 이웃에 대해 동일한 확률로 샘플링

**중요도 기반 샘플링:**
- 엣지 가중치나 노드 중요도를 고려한 샘플링

**계층적 샘플링:**
- 레이어별로 다른 샘플링 크기 적용

### 2.7. 손실 함수

**비지도 학습 (Unsupervised):**
$$J_G(\mathbf{z}_u) = -\log(\sigma(\mathbf{z}_u^T \mathbf{z}_v)) - Q \cdot \mathbb{E}_{v_n \sim P_n(v)} \log(\sigma(-\mathbf{z}_u^T \mathbf{z}_{v_n}))$$

여기서:
- $(u,v)$: 그래프에서 인접한 노드 쌍
- $P_n$: 네거티브 샘플링 분포
- $Q$: 네거티브 샘플 수

### 2.8. 특성 비교

| 특성 | GCN | GraphSAGE |
|------|-----|-----------|
| **학습 방식** | Transductive | Inductive |
| **확장성** | 제한적 | 우수 |
| **메모리 사용** | $O(n \times d)$ | $O(S^L \times d)$ |
| **새 노드 처리** | 불가능 | 가능 |
| **집계 함수** | 고정 (평균) | 다양함 |

### 2.9. 구현 예시

```python
class GraphSAGELayer(nn.Module):
    def __init__(self, in_features, out_features, aggregator='mean'):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.aggregator = aggregator
        
        # 자기 자신과 이웃을 결합하는 레이어
        self.linear = nn.Linear(in_features * 2, out_features)
        
        if aggregator == 'lstm':
            self.lstm = nn.LSTM(in_features, in_features, batch_first=True)
        elif aggregator == 'pool':
            self.pool_linear = nn.Linear(in_features, in_features)
            
    def forward(self, x, edge_index):
        # 이웃 집계
        if self.aggregator == 'mean':
            neighbor_emb = self.aggregate_mean(x, edge_index)
        elif self.aggregator == 'lstm':
            neighbor_emb = self.aggregate_lstm(x, edge_index)
        elif self.aggregator == 'pool':
            neighbor_emb = self.aggregate_pool(x, edge_index)
            
        # 자기 자신과 이웃 정보 결합
        self_emb = x
        combined = torch.cat([self_emb, neighbor_emb], dim=1)
        
        # 선형 변환 및 활성화
        out = F.relu(self.linear(combined))
        
        # L2 정규화
        out = F.normalize(out, p=2, dim=1)
        
        return out
```

### 2.10. 응용 및 확장

**산업 응용:**
- 추천 시스템 (Pinterest, Uber)
- 소셜 네트워크 분석
- 지식 그래프 임베딩

**확장 연구:**
- FastGraphSAGE: 고속 근사 알고리즘
- GraphSAINT: 서브그래프 샘플링
- Control Variate: 분산 감소 기법

---

## 3. GAT (Graph Attention Network)

### 3.1. 개념 및 동기
GAT는 **Transformer의 어텐션 메커니즘**을 그래프에 적용한 모델로, 이웃 노드들에 대해 **적응적 중요도**를 학습합니다.

### 3.2. 핵심 아이디어

**문제점:**
- GCN: 모든 이웃에 동일한 가중치 적용
- GraphSAGE: 집계 함수는 다양하지만 여전히 균등한 중요도

**해결책:**
- 노드 쌍별로 동적 어텐션 가중치 계산
- 중요한 이웃에 더 큰 가중치 부여

### 3.3. 어텐션 메커니즘

**1. 선형 변환:**
$$\mathbf{h}'_i = \mathbf{W} \mathbf{h}_i$$

**2. 어텐션 계수 계산:**
$$e_{ij} = a(\mathbf{W}\mathbf{h}_i, \mathbf{W}\mathbf{h}_j)$$

**3. 어텐션 함수 (일반적 형태):**
$$e_{ij} = \text{LeakyReLU}(\mathbf{a}^T [\mathbf{W}\mathbf{h}_i || \mathbf{W}\mathbf{h}_j])$$

여기서:
- $\mathbf{a} \in \mathbb{R}^{2F'}$: 학습 가능한 어텐션 파라미터
- $||$: 연결(concatenation) 연산
- $\mathbf{W} \in \mathbb{R}^{F' \times F}$: 공유 선형 변환 행렬

**4. 어텐션 가중치 정규화:**
$$\alpha_{ij} = \text{softmax}_j(e_{ij}) = \frac{\exp(e_{ij})}{\sum_{k \in \mathcal{N}_i} \exp(e_{ik})}$$

**5. 최종 출력:**
$$\mathbf{h}'_i = \sigma\left(\sum_{j \in \mathcal{N}_i} \alpha_{ij} \mathbf{W} \mathbf{h}_j\right)$$

### 3.4. 멀티헤드 어텐션

**K개의 어텐션 헤드:**
$$\mathbf{h}'_i = \parallel_{k=1}^K \sigma\left(\sum_{j \in \mathcal{N}_i} \alpha_{ij}^k \mathbf{W}^k \mathbf{h}_j\right)$$

**최종 레이어 (평균화):**
$$\mathbf{h}'_i = \sigma\left(\frac{1}{K} \sum_{k=1}^K \sum_{j \in \mathcal{N}_i} \alpha_{ij}^k \mathbf{W}^k \mathbf{h}_j\right)$$

### 3.5. 알고리즘 상세

```python
def gat_attention(h_i, h_j, W, a):
    """
    GAT 어텐션 계산
    Args:
        h_i, h_j: 노드 특징 벡터
        W: 선형 변환 행렬
        a: 어텐션 파라미터
    """
    # 1. 선형 변환
    Wh_i = torch.matmul(h_i, W)
    Wh_j = torch.matmul(h_j, W)
    
    # 2. 연결
    concat = torch.cat([Wh_i, Wh_j], dim=1)
    
    # 3. 어텐션 계수
    e_ij = F.leaky_relu(torch.matmul(concat, a))
    
    return e_ij
```

### 3.6. GAT vs 다른 어텐션 메커니즘

| 특성 | Transformer | GAT | Graph Transformer |
|------|-------------|-----|-------------------|
| **입력** | 시퀀스 | 그래프 | 그래프 |
| **어텐션 범위** | 전체 시퀀스 | 직접 이웃 | 전체 그래프 |
| **위치 인코딩** | 필요 | 불필요 | 필요 |
| **복잡도** | $O(n^2)$ | $O(|E|)$ | $O(n^2)$ |

### 3.7. 마스킹 및 정규화

**마스킹:**
- 연결되지 않은 노드 쌍에 $-\infty$ 할당
- Softmax 후 0이 되어 어텐션 무시

**드롭아웃:**
$$\alpha_{ij} = \text{Dropout}(\alpha_{ij})$$

### 3.8. 구현 예시

```python
class GATLayer(nn.Module):
    def __init__(self, in_features, out_features, dropout=0.6, alpha=0.2, concat=True):
        super(GATLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.dropout = dropout
        self.alpha = alpha
        self.concat = concat
        
        # 선형 변환 가중치
        self.W = nn.Parameter(torch.empty(size=(in_features, out_features)))
        # 어텐션 파라미터
        self.a = nn.Parameter(torch.empty(size=(2*out_features, 1)))
        
        self.leakyrelu = nn.LeakyReLU(self.alpha)
        self.reset_parameters()
        
    def reset_parameters(self):
        nn.init.xavier_uniform_(self.W.data, gain=1.414)
        nn.init.xavier_uniform_(self.a.data, gain=1.414)
        
    def forward(self, h, adj):
        # 선형 변환
        Wh = torch.mm(h, self.W)  # [N, out_features]
        
        # 어텐션 계수 계산
        e = self._prepare_attentional_mechanism_input(Wh)
        
        # 마스킹 (연결되지 않은 노드는 -inf)
        zero_vec = -9e15*torch.ones_like(e)
        attention = torch.where(adj > 0, e, zero_vec)
        
        # Softmax
        attention = F.softmax(attention, dim=1)
        attention = F.dropout(attention, self.dropout, training=self.training)
        
        # 가중합
        h_prime = torch.matmul(attention, Wh)
        
        if self.concat:
            return F.elu(h_prime)
        else:
            return h_prime
            
    def _prepare_attentional_mechanism_input(self, Wh):
        # Wh: [N, out_features]
        N = Wh.size()[0]
        
        # 모든 노드 쌍에 대해 연결
        Wh_repeated_in_chunks = Wh.repeat_interleave(N, dim=0)
        Wh_repeated_alternating = Wh.repeat(N, 1)
        
        # 연결
        all_combinations_matrix = torch.cat([Wh_repeated_in_chunks, Wh_repeated_alternating], dim=1)
        
        # 어텐션 계산
        return self.leakyrelu(torch.matmul(all_combinations_matrix, self.a)).view(N, N)
```

### 3.9. 장단점

**장점:**
- **적응적 중요도**: 노드별로 이웃의 중요도를 다르게 학습
- **Inductive 학습**: 새로운 노드에 적용 가능
- **해석 가능성**: 어텐션 가중치를 통한 시각화 가능
- **안정적 학습**: GCN보다 깊은 네트워크 구성 가능

**단점:**
- **계산 복잡도**: 어텐션 계산으로 인한 오버헤드
- **메모리 사용량**: 어텐션 가중치 저장 필요
- **과적합 위험**: 파라미터 수 증가

### 3.10. 확장 및 변형

**GAT v2:**
- 동적 어텐션 함수
- 더 나은 표현력

**SuperGAT:**
- 어텐션 정규화
- Edge prediction 보조 태스크

**GATv2:**
$$e_{ij} = \mathbf{a}^T \text{LeakyReLU}(\mathbf{W} [\mathbf{h}_i || \mathbf{h}_j])$$

---

## 4. Graph Transformer

### 4.1. 개념
**Graph Transformer**는 Transformer 아키텍처를 그래프에 완전히 적용한 모델로, 모든 노드 쌍 간의 어텐션을 계산합니다.

### 4.2. 핵심 구성요소

**1. 위치 인코딩 (Positional Encoding):**
- **라플라시안 고유벡터**: $\mathbf{PE}_i = [\lambda_1 u_1(i), \lambda_2 u_2(i), ..., \lambda_k u_k(i)]$
- **Random Walk**: 노드 간 Random Walk 거리 기반
- **Central Encoding**: 중심성 기반 위치 정보

**2. 구조적 어텐션:**
$$A_{ij} = \text{softmax}\left(\frac{(\mathbf{h}_i \mathbf{W}_Q)(\mathbf{h}_j \mathbf{W}_K)^T + b_{ij}}{\sqrt{d}}\right)$$

여기서 $b_{ij}$는 구조적 편향(structural bias)

### 4.3. GraphiT 아키텍처

**입력 임베딩:**
$$\mathbf{h}_i^{(0)} = \mathbf{x}_i \mathbf{W}_x + \mathbf{PE}_i \mathbf{W}_{PE}$$

**어텐션 계산:**
$$\text{Attention}(\mathbf{Q}, \mathbf{K}, \mathbf{V}) = \text{softmax}\left(\frac{\mathbf{Q}\mathbf{K}^T + \mathbf{B}}{\sqrt{d}}\right)\mathbf{V}$$

**구조적 편향 행렬:**
$$\mathbf{B}_{ij} = \phi(SP(i,j))$$

여기서 $SP(i,j)$는 최단 경로 거리

---

## 5. GIN (Graph Isomorphism Network)

### 5.1. 이론적 동기
GIN은 **Weisfeiler-Lehman (WL) 테스트**의 표현력을 최대한 달성하도록 설계된 이론적으로 탄탄한 모델입니다.

### 5.2. 핵심 정리

**정리 (Xu et al., 2019):**
GNN이 WL-테스트만큼 강력하려면, 집계 함수가 **단사 함수(injective)**여야 합니다.

**단사 집계 함수:**
$$f\left((1+\epsilon) \cdot \mathbf{h}_v + \sum_{u \in \mathcal{N}(v)} \mathbf{h}_u\right)$$

### 5.3. GIN 레이어

**업데이트 규칙:**
$$\mathbf{h}_v^{(k+1)} = \text{MLP}^{(k)}\left((1+\epsilon^{(k)}) \cdot \mathbf{h}_v^{(k)} + \sum_{u \in \mathcal{N}(v)} \mathbf{h}_u^{(k)}\right)$$

**학습 가능한 $\epsilon$:**
$$\epsilon^{(k)} = \text{학습 파라미터 또는 고정값}$$

### 5.4. 그래프 수준 예측

**그래프 표현:**
$$\mathbf{h}_G = \text{CONCAT}\left(\text{READOUT}\left(\left\{\mathbf{h}_v^{(k)} | v \in G\right\}\right) | k = 0, 1, ..., K\right)$$

**READOUT 함수:**
- SUM: $\sum_{v \in G} \mathbf{h}_v^{(k)}$
- MEAN: $\frac{1}{|V|} \sum_{v \in G} \mathbf{h}_v^{(k)}$
- MAX: $\max_{v \in G} \mathbf{h}_v^{(k)}$

---

## 6. 아키텍처 비교 및 선택 가이드

### 6.1. 성능 비교표

| 모델 | 이론적 기반 | 표현력 | 확장성 | Inductive | 해석가능성 |
|------|-------------|--------|--------|-----------|------------|
| **GCN** | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐ | ❌ | ⭐⭐ |
| **GraphSAGE** | ⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐ |
| **GAT** | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ |
| **Graph Transformer** | ⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐ |
| **GIN** | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐ |

### 6.2. 태스크별 추천

**노드 분류:**
- 작은 그래프: GCN, GAT
- 큰 그래프: GraphSAGE
- 높은 성능 필요: GAT, Graph Transformer

**그래프 분류:**
- 이론적 보장: GIN
- 실용성: GraphSAGE + global pooling
- 해석가능성: GAT

**링크 예측:**
- 효율성: GCN
- 성능: GAT
- 대규모: GraphSAGE

### 6.3. 하이브리드 접근법

**계층적 구조:**
```python
class HybridGNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = GCNLayer(input_dim, hidden_dim)
        self.layer2 = GATLayer(hidden_dim, hidden_dim)
        self.layer3 = GraphSAGELayer(hidden_dim, output_dim)
    
    def forward(self, x, edge_index):
        x = self.layer1(x, edge_index)
        x = self.layer2(x, edge_index)
        x = self.layer3(x, edge_index)
        return x
```

각 GNN 아키텍처는 고유한 강점과 적용 분야를 가지므로, 문제의 특성과 요구사항에 따라 적절한 모델을 선택하거나 하이브리드 접근법을 사용하는 것이 중요합니다.
