# 트랜스포머 (Transformer) 아키텍처

**트랜스포머(Transformer)**는 2017년 구글의 논문 "Attention Is All You Need"에서 처음 제안된 모델로, 기존의 순환 신경망(RNN) 구조를 완전히 배제하고 **어텐션(Attention) 메커니즘**만으로 시퀀스 데이터를 처리하는 혁신적인 아키텍처입니다.

트랜스포머는 병렬 처리의 이점을 극대화하여 학습 속도를 크게 향상시켰고, 자연어 처리(NLP) 분야에서 전례 없는 성능을 달성하며 현대 AI 모델의 기반이 되었습니다.

## 1. 트랜스포머의 전체 구조: 인코더-디코더

트랜스포머는 Seq2Seq 모델과 마찬가지로, 입력을 처리하는 **인코더(Encoder)**와 출력을 생성하는 **디코더(Decoder)** 스택으로 구성됩니다.
- **인코더 스택:** 여러 개의 동일한 인코더 레이어를 쌓은 구조. 입력 시퀀스의 각 단어에 대한 문맥적인 표현(representation)을 생성합니다.
- **디코더 스택:** 여러 개의 동일한 디코더 레이어를 쌓은 구조. 인코더의 출력과 이전에 생성된 출력 단어들을 입력으로 받아, 다음 단어를 예측합니다.

![Transformer Architecture](https://i.imgur.com/3mMLKx5.png)

## 2. 핵심 구성 요소

### 2.1. 셀프 어텐션 (Self-Attention)

**개념:**
"자기 자신"에게 어텐션을 적용한다는 의미로, **하나의 시퀀스 내에서** 단어들 간의 관계와 의존성을 직접적으로 계산하여 문맥을 파악하는 메커니즘입니다.

**예시:**
"The animal didn't cross the street because **it** was too tired." 라는 문장에서, 'it'이 'animal'을 가리키는 것인지 'street'을 가리키는 것인지를 파악하기 위해, 'it'과 문장 내 다른 모든 단어 간의 연관성을 계산합니다.

**수학적 정의:**
주어진 입력 시퀀스 $\mathbf{X} \in \mathbb{R}^{n \times d_{model}}$에 대해, Query, Key, Value 행렬은 다음과 같이 계산됩니다:

$$\mathbf{Q} = \mathbf{X}\mathbf{W}^Q, \quad \mathbf{K} = \mathbf{X}\mathbf{W}^K, \quad \mathbf{V} = \mathbf{X}\mathbf{W}^V$$

여기서:
- $\mathbf{W}^Q, \mathbf{W}^K, \mathbf{W}^V \in \mathbb{R}^{d_{model} \times d_k}$: 학습 가능한 가중치 행렬
- $n$: 시퀀스 길이
- $d_{model}$: 모델 차원
- $d_k$: 키/쿼리 벡터의 차원

**어텐션 계산:**
$$\text{Attention}(\mathbf{Q}, \mathbf{K}, \mathbf{V}) = \text{softmax}\left(\frac{\mathbf{Q}\mathbf{K}^T}{\sqrt{d_k}}\right)\mathbf{V}$$

**단계별 과정:**

1. **Query, Key, Value 생성:**
   각 단어(입력 임베딩)에 대해 세 가지 다른 벡터를 생성합니다:
   - **Query (Q):** 현재 단어를 나타내는 '질문' 벡터
   - **Key (K):** 시퀀스 내의 모든 단어들을 나타내는 '키' 벡터
   - **Value (V):** 시퀀스 내의 모든 단어들의 실제 '내용'을 담는 벡터

2. **어텐션 점수 계산:**
   $$e_{ij} = \frac{\mathbf{q}_i \cdot \mathbf{k}_j}{\sqrt{d_k}}$$
   현재 위치 $i$의 Query와 모든 위치 $j$의 Key 간의 유사도를 계산합니다.

3. **어텐션 가중치 정규화:**
   $$\alpha_{ij} = \frac{\exp(e_{ij})}{\sum_{k=1}^n \exp(e_{ik})}$$
   Softmax를 적용하여 가중치의 합이 1이 되도록 정규화합니다.

4. **가중합 계산:**
   $$\mathbf{o}_i = \sum_{j=1}^n \alpha_{ij} \mathbf{v}_j$$
   어텐션 가중치를 Value 벡터에 곱하여 최종 출력을 생성합니다.

**스케일링의 필요성:**
$\sqrt{d_k}$로 나누는 이유는 내적 값이 차원에 비례하여 커지면 softmax 함수의 그래디언트가 매우 작아지는 문제를 방지하기 위함입니다.

### 2.2. 멀티-헤드 어텐션 (Multi-Head Attention)

**개념:**
하나의 셀프 어텐션을 수행하는 대신, 여러 개의 "헤드(head)"가 **병렬적으로** 셀프 어텐션을 수행하는 구조입니다.

**수학적 정의:**
$$\text{MultiHead}(\mathbf{Q}, \mathbf{K}, \mathbf{V}) = \text{Concat}(\text{head}_1, ..., \text{head}_h)\mathbf{W}^O$$

여기서 각 헤드는:
$$\text{head}_i = \text{Attention}(\mathbf{Q}\mathbf{W}_i^Q, \mathbf{K}\mathbf{W}_i^K, \mathbf{V}\mathbf{W}_i^V)$$

**파라미터:**
- $\mathbf{W}_i^Q, \mathbf{W}_i^K, \mathbf{W}_i^V \in \mathbb{R}^{d_{model} \times d_k}$: 각 헤드의 투영 행렬
- $\mathbf{W}^O \in \mathbb{R}^{h \times d_k \times d_{model}}$: 출력 투영 행렬
- $h$: 헤드의 수 (일반적으로 8 또는 16)
- $d_k = d_{model} / h$: 각 헤드의 차원

**동작 방식:**

1. **차원 분할:**
   입력을 $h$개의 헤드로 나누어 각각 $d_k$ 차원을 갖도록 합니다.

2. **병렬 어텐션 계산:**
   각 헤드는 독립적으로 어텐션을 계산하여 서로 다른 표현 부공간(representation subspace)에서의 정보를 캡처합니다:
   - 헤드 1: 문법적 관계 (주어-동사, 수식 관계)
   - 헤드 2: 의미적 관계 (주제, 감정)
   - 헤드 3: 장거리 의존성
   - ...

3. **결과 결합:**
   ```python
   # 의사 코드
   heads = []
   for i in range(h):
       head_i = attention(Q @ W_Q[i], K @ W_K[i], V @ W_V[i])
       heads.append(head_i)
   
   output = concat(heads) @ W_O
   ```

4. **최종 선형 변환:**
   연결된 헤드들을 원래 $d_{model}$ 차원으로 다시 투영합니다.

**효과 및 이점:**
- **다양한 관점:** 각 헤드가 서로 다른 종류의 관계를 학습
- **표현력 증가:** 여러 표현 부공간에서의 정보 결합
- **안정적 학습:** 단일 어텐션보다 더 안정적인 그래디언트
- **해석 가능성:** 각 헤드가 담당하는 언어적 기능 분석 가능

**실제 구현 예시:**
```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_heads):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        
        self.W_Q = nn.Linear(d_model, d_model)
        self.W_K = nn.Linear(d_model, d_model)
        self.W_V = nn.Linear(d_model, d_model)
        self.W_O = nn.Linear(d_model, d_model)
        
    def forward(self, query, key, value, mask=None):
        batch_size, seq_len, _ = query.size()
        
        # Q, K, V 계산 및 헤드별 분할
        Q = self.W_Q(query).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        K = self.W_K(key).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        V = self.W_V(value).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        
        # 각 헤드에서 어텐션 계산
        attention_output = self.attention(Q, K, V, mask)
        
        # 헤드 결합
        attention_output = attention_output.transpose(1, 2).contiguous().view(
            batch_size, seq_len, self.d_model
        )
        
        return self.W_O(attention_output)
    
    def attention(self, Q, K, V, mask=None):
        scores = torch.matmul(Q, K.transpose(-2, -1)) / torch.sqrt(torch.tensor(self.d_k))
        
        if mask is not None:
            scores.masked_fill_(mask == 0, -1e9)
        
        attention_weights = F.softmax(scores, dim=-1)
        return torch.matmul(attention_weights, V)
```

### 2.3. 포지셔널 인코딩 (Positional Encoding)

**문제점:**
트랜스포머는 RNN과 달리 순환 구조가 없기 때문에, 단어의 순서 정보를 자체적으로 알 수 없습니다. "I go to school"과 "school go I to"를 동일하게 처리하는 문제가 발생합니다.

**해결책:**
**포지셔널 인코딩(Positional Encoding)**은 각 단어의 **위치 정보**를 담고 있는 벡터를 만들어 입력 임베딩에 더해주는 방식입니다.

**수학적 정의:**
원문에서 제안한 사인파 기반 포지셔널 인코딩:

$$PE_{(pos, 2i)} = \sin\left(\frac{pos}{10000^{2i/d_{model}}}\right)$$
$$PE_{(pos, 2i+1)} = \cos\left(\frac{pos}{10000^{2i/d_{model}}}\right)$$

여기서:
- $pos$: 위치 (0, 1, 2, ..., max_seq_len-1)
- $i$: 차원 인덱스 (0, 1, 2, ..., $d_{model}/2-1$)
- $d_{model}$: 모델 차원

**특성 및 장점:**

1. **고유성:** 각 위치마다 고유한 인코딩 벡터
2. **상대적 위치:** $PE_{pos+k}$는 $PE_{pos}$의 선형 함수로 표현 가능
3. **확장성:** 학습 시보다 긴 시퀀스에도 적용 가능
4. **주기성:** 서로 다른 주기의 사인파 조합으로 패턴 다양성 확보

**직관적 이해:**
- **짝수 차원:** 사인 함수 사용, 빠른 변화
- **홀수 차원:** 코사인 함수 사용, 느린 변화
- 낮은 차원: 높은 주파수 (빠른 위치 변화 감지)
- 높은 차원: 낮은 주파수 (넓은 범위 위치 관계 학습)

**구현 예시:**
```python
import torch
import math

def positional_encoding(max_seq_len, d_model):
    pe = torch.zeros(max_seq_len, d_model)
    
    for pos in range(max_seq_len):
        for i in range(0, d_model, 2):
            pe[pos, i] = math.sin(pos / (10000 ** (i / d_model)))
            if i + 1 < d_model:
                pe[pos, i + 1] = math.cos(pos / (10000 ** (i / d_model)))
    
    return pe

# 또는 벡터화된 구현
def positional_encoding_vectorized(max_seq_len, d_model):
    pe = torch.zeros(max_seq_len, d_model)
    position = torch.arange(0, max_seq_len).unsqueeze(1).float()
    
    div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                        -(math.log(10000.0) / d_model))
    
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    
    return pe
```

**대안적 방법들:**

1. **학습 가능한 포지셔널 임베딩:**
   ```python
   self.pos_embedding = nn.Embedding(max_seq_len, d_model)
   ```

2. **상대적 포지셔널 인코딩 (T5, DeBERTa):**
   - 절대 위치가 아닌 상대적 거리 정보 사용
   - 더 나은 일반화 성능

3. **RoPE (Rotary Positional Embedding):**
   - Query와 Key에 회전 변환 적용
   - 최근 모델들에서 널리 사용

**최종 입력 계산:**
$$\text{Input} = \text{TokenEmbedding} + \text{PositionalEncoding}$$

포지셔널 인코딩 덕분에 모델은 "나는 학교에 간다"와 "간다 학교에 나는"을 구별할 수 있게 됩니다.

### 2.4. 잔차 연결 및 층 정규화 (Add & Norm)

각 서브 레이어(Multi-Head Attention, Feed-Forward Network)의 출력은 **잔차 연결(Residual Connection)**과 **층 정규화(Layer Normalization)**를 거칩니다.

**수식적 표현:**
$$\text{output} = \text{LayerNorm}(\mathbf{x} + \text{SubLayer}(\mathbf{x}))$$

**구성 요소:**

**1. 잔차 연결 (Residual Connection):**
- ResNet에서 제안된 스킵 연결과 동일한 개념
- 입력 $\mathbf{x}$를 서브 레이어 출력에 직접 더함
- **목적:** 깊은 네트워크에서도 그래디언트가 잘 흐르도록 보장

$$\mathbf{y} = \mathbf{x} + F(\mathbf{x})$$

여기서 $F(\mathbf{x})$는 서브 레이어 함수입니다.

**2. 층 정규화 (Layer Normalization):**
각 레이어의 출력을 정규화하여 학습을 안정화합니다.

$$\text{LayerNorm}(\mathbf{x}) = \gamma \odot \frac{\mathbf{x} - \mu}{\sigma} + \beta$$

여기서:
- $\mu = \frac{1}{d} \sum_{i=1}^{d} x_i$ (평균)
- $\sigma = \sqrt{\frac{1}{d} \sum_{i=1}^{d} (x_i - \mu)^2}$ (표준편차)
- $\gamma, \beta \in \mathbb{R}^d$ (학습 가능한 파라미터)
- $\odot$ (원소별 곱셈)

**Layer Norm vs Batch Norm 비교:**

| 특성 | Layer Norm | Batch Norm |
|------|------------|------------|
| **정규화 축** | Feature 차원 | Batch 차원 |
| **의존성** | 샘플 독립적 | 배치 의존적 |
| **시퀀스 데이터** | 적합 | 부적합 |
| **추론 시** | 동일 | 이동 평균 사용 |

**장점 및 효과:**

1. **그래디언트 흐름 개선:** 잔차 연결로 그래디언트 소실 방지
2. **학습 안정성:** Layer Norm으로 내부 공변량 이동 감소
3. **수렴 속도:** 더 빠른 학습 수렴
4. **일반화:** 과적합 방지 효과

**구현 예시:**
```python
class ResidualConnection(nn.Module):
    def __init__(self, d_model, dropout=0.1):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, sublayer):
        """
        잔차 연결과 층 정규화 적용
        Original: LayerNorm(x + sublayer(x))
        """
        return x + self.dropout(sublayer(self.norm(x)))
        # 주: 일부 구현에서는 Post-Norm 대신 Pre-Norm 사용
```

**Pre-Norm vs Post-Norm:**
- **Post-Norm:** $\text{LayerNorm}(\mathbf{x} + \text{SubLayer}(\mathbf{x}))$ (원논문)
- **Pre-Norm:** $\mathbf{x} + \text{SubLayer}(\text{LayerNorm}(\mathbf{x}))$ (최근 선호)

Pre-Norm이 더 안정적인 학습을 제공하여 최근 모델들에서 널리 사용됩니다.

### 2.5. 피드 포워드 신경망 (Position-wise Feed-Forward Networks)

각 인코더와 디코더 레이어에는 멀티-헤드 어텐션 이후에 **위치별 피드포워드 신경망(Position-wise FFN)**이 위치합니다.

**수학적 정의:**
$$\text{FFN}(\mathbf{x}) = \max(0, \mathbf{x}\mathbf{W}_1 + \mathbf{b}_1)\mathbf{W}_2 + \mathbf{b}_2$$

또는 더 일반적으로:
$$\text{FFN}(\mathbf{x}) = \text{GELU}(\mathbf{x}\mathbf{W}_1 + \mathbf{b}_1)\mathbf{W}_2 + \mathbf{b}_2$$

**파라미터:**
- $\mathbf{W}_1 \in \mathbb{R}^{d_{model} \times d_{ff}}$: 첫 번째 레이어 가중치
- $\mathbf{W}_2 \in \mathbb{R}^{d_{ff} \times d_{model}}$: 두 번째 레이어 가중치
- $\mathbf{b}_1 \in \mathbb{R}^{d_{ff}}, \mathbf{b}_2 \in \mathbb{R}^{d_{model}}$: 편향
- $d_{ff} = 4 \times d_{model}$ (일반적인 설정)

**구조적 특징:**

1. **위치별 독립성:** 각 위치에서 동일한 FFN이 독립적으로 적용
2. **차원 확장 및 축소:** 
   - $d_{model} \rightarrow d_{ff} \rightarrow d_{model}$
   - 중간층에서 차원을 4배로 확장 후 다시 축소
3. **비선형성:** 활성화 함수를 통한 비선형 변환

**활성화 함수 선택:**

**ReLU (원논문):**
$$\text{ReLU}(x) = \max(0, x)$$

**GELU (최근 선호):**
$$\text{GELU}(x) = x \cdot \Phi(x) = x \cdot \frac{1}{2}\left[1 + \text{erf}\left(\frac{x}{\sqrt{2}}\right)\right]$$

**Swish/SiLU:**
$$\text{SiLU}(x) = x \cdot \text{sigmoid}(x)$$

**역할 및 중요성:**

1. **표현력 증가:** 어텐션 결과를 더 복잡한 비선형 변환으로 처리
2. **특징 변환:** 각 위치에서 독립적인 특징 변환 수행
3. **모델 용량:** 전체 파라미터의 상당 부분을 차지 (약 2/3)

**구현 예시:**
```python
class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super().__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.GELU()  # 또는 nn.ReLU()
        
    def forward(self, x):
        return self.w_2(self.dropout(self.activation(self.w_1(x))))

# 또는 더 간단하게
class FFN(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super().__init__()
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout)
        )
    
    def forward(self, x):
        return self.ffn(x)
```

**변형들:**

1. **GLU (Gated Linear Unit):**
   $$\text{GLU}(x) = (x\mathbf{W}_1 + \mathbf{b}_1) \odot \sigma(x\mathbf{W}_2 + \mathbf{b}_2)$$

2. **SwiGLU (GPT-3에서 사용):**
   $$\text{SwiGLU}(x) = \text{Swish}(x\mathbf{W}_1)\odot (x\mathbf{W}_2)$$

3. **Mixture of Experts (MoE):**
   - 여러 FFN 전문가 중 일부만 활성화
   - 모델 용량 증가 without 계산량 증가

FFN은 어텐션 메커니즘과 함께 트랜스포머의 핵심 구성요소로, 각 위치에서 독립적인 비선형 변환을 통해 모델의 표현 능력을 크게 향상시킵니다.

## 3. 인코더와 디코더의 세부 구조

### 3.1. 인코더 레이어
각 인코더 레이어는 다음과 같이 구성됩니다:
1. **멀티-헤드 셀프 어텐션:** 입력 시퀀스 내 단어들 간의 관계 학습
2. **Add & Norm:** 잔차 연결 및 정규화
3. **피드 포워드 신경망:** 비선형 변환
4. **Add & Norm:** 다시 한 번 잔차 연결 및 정규화

### 3.2. 디코더 레이어
각 디코더 레이어는 다음과 같이 구성됩니다:

**구조:**
1. **마스크드 셀프 어텐션:** 이미 생성된 토큰들 간의 관계 학습 (미래 정보 차단)
2. **Add & Norm**
3. **인코더-디코더 어텐션:** 인코더 출력과 디코더 상태 간의 관계 학습
4. **Add & Norm**
5. **피드 포워드 신경망**
6. **Add & Norm**

### 3.3. 마스킹 (Masking)

트랜스포머에서는 두 가지 주요 마스킹 기법이 사용됩니다.

**1. 패딩 마스크 (Padding Mask):**
- **목적:** 시퀀스 길이가 다를 때 패딩 토큰에 어텐션이 가지 않도록 차단
- **적용:** 모든 어텐션 레이어에서 사용
- **구현:**
  ```python
  def create_padding_mask(seq, pad_token_id=0):
      mask = (seq != pad_token_id).unsqueeze(1).unsqueeze(2)
      return mask  # shape: (batch_size, 1, 1, seq_len)
  ```

**2. 룩어헤드 마스크 (Look-ahead Mask/Causal Mask):**
- **목적:** 디코더에서 미래 토큰 정보를 보지 못하도록 차단
- **적용:** 디코더의 마스크드 셀프 어텐션에서만 사용
- **수학적 표현:**

$$\text{Mask}_{ij} = \begin{cases} 
0 & \text{if } i < j \\
-\infty & \text{if } i \geq j
\end{cases}$$

**구현:**
```python
def create_look_ahead_mask(size):
    mask = torch.triu(torch.ones(size, size), diagonal=1)
    return mask == 1  # True인 위치를 -inf로 마스킹

# 예시: size=4인 경우
# [[False, True,  True,  True ],
#  [False, False, True,  True ],
#  [False, False, False, True ],
#  [False, False, False, False]]
```

**어텐션에서의 마스킹 적용:**
```python
def masked_attention(Q, K, V, mask=None):
    scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(d_k)
    
    if mask is not None:
        scores.masked_fill_(mask == 0, -1e9)  # 매우 작은 값으로 마스킹
    
    attention_weights = F.softmax(scores, dim=-1)
    return torch.matmul(attention_weights, V)
```

**마스킹의 효과:**
- Softmax 적용 시 마스킹된 위치는 0에 가까운 확률을 가짐
- 결과적으로 해당 위치의 정보가 최종 출력에 영향을 주지 않음

### 3.4. 인코더-디코더 어텐션 (Cross-Attention)

디코더의 두 번째 어텐션 레이어에서 사용되는 특별한 어텐션입니다.

**구조:**
- **Query:** 디코더의 이전 레이어 출력
- **Key, Value:** 인코더의 최종 출력
- **목적:** 디코더가 인코더의 모든 위치에 어텐션을 적용하여 소스 정보 활용

**수식:**
$$\text{CrossAttention} = \text{Attention}(\mathbf{Q}_{\text{dec}}, \mathbf{K}_{\text{enc}}, \mathbf{V}_{\text{enc}})$$

여기서:
- $\mathbf{Q}_{\text{dec}}$: 디코더에서 생성된 Query
- $\mathbf{K}_{\text{enc}}, \mathbf{V}_{\text{enc}}$: 인코더에서 생성된 Key, Value

**실제 동작 예시 (기계 번역):**
- 영어 "The cat sat" → 프랑스어 "Le chat"
- "Le" 생성 시: 인코더의 "The" 위치에 높은 어텐션
- "chat" 생성 시: 인코더의 "cat" 위치에 높은 어텐션

이러한 어텐션 메커니즘을 통해 디코더는 소스 시퀀스의 관련 부분에 적절히 집중할 수 있습니다.

## 4. 트랜스포머의 주요 장점

### 4.1. 병렬 처리 가능
**RNN의 한계:**
- 순차적 처리로 인한 병렬화 불가
- 시간 복잡도: $O(n)$ (순차 처리)

**트랜스포머의 해결:**
- 모든 위치에서 동시에 어텐션 계산
- 시간 복잡도: $O(1)$ (병렬 처리)
- GPU 활용도 극대화

### 4.2. 장기 의존성 문제 해결
**RNN의 문제:**
- 그래디언트 소실/폭발로 장거리 의존성 학습 어려움
- 정보 병목 현상 (hidden state 크기 제한)

**트랜스포머의 해결:**
- 직접적인 어텐션 연결로 모든 위치 간 관계 모델링
- 경로 길이: $O(1)$ vs RNN의 $O(n)$
- 잔차 연결을 통한 안정적인 그래디언트 흐름

### 4.3. 유연한 표현 학습
**멀티헤드 어텐션의 효과:**
- 다양한 표현 부공간에서 관계 학습
- 문법적, 의미적, 구조적 정보 동시 캡처
- 해석 가능한 어텐션 패턴

### 4.4. 확장성 (Scalability)
**모델 크기 확장:**
- 레이어 수, 차원 수 쉽게 조정 가능
- 대규모 모델에서도 안정적 학습
- 모델 크기에 따른 성능 향상 (Scaling Laws)

### 4.5. 전이 학습 친화적
**사전 훈련 패러다임:**
- 대규모 무라벨 데이터로 언어 모델 학습
- 다양한 다운스트림 태스크로 미세조정
- 효율적인 지식 전이

## 5. 계산 복잡도 분석

### 5.1. 시간 복잡도
| 연산 | 시퀀스 길이 | 메모리 | 병렬 처리 |
|------|-------------|--------|-----------|
| **Self-Attention** | $O(n^2 \cdot d)$ | $O(n^2)$ | $O(1)$ |
| **RNN** | $O(n \cdot d^2)$ | $O(d)$ | $O(n)$ |
| **CNN (1D)** | $O(n \cdot d^2 \cdot k)$ | $O(d)$ | $O(1)$ |

### 5.2. 공간 복잡도
**어텐션 행렬:** $O(n^2)$ - 긴 시퀀스에서 메모리 병목
**해결 방안:**
- Sparse Attention (Longformer, BigBird)
- Linear Attention (Performer, LinearFormer)
- Hierarchical Attention

## 6. 한계 및 도전과제

### 6.1. 제곱 복잡도
- 시퀀스 길이 $n$에 대해 $O(n^2)$ 메모리 및 계산
- 긴 시퀀스 처리 시 확장성 문제

### 6.2. 위치 정보의 한계
- 절대적 위치 인코딩의 일반화 문제
- 시퀀스 길이 외삽 (Extrapolation) 어려움

### 6.3. 귀납적 편향 부족
- CNN의 지역성, RNN의 순차성 같은 구조적 편향 없음
- 더 많은 데이터와 컴퓨팅 자원 필요

## 7. 최신 발전 방향

### 7.1. 효율성 개선
- **Sparse Attention:** 어텐션을 희소하게 만들어 복잡도 감소
- **Linear Attention:** 선형 복잡도의 근사 어텐션
- **MoE (Mixture of Experts):** 조건부 계산으로 효율성 증대

### 7.2. 구조 개선
- **Pre-Norm vs Post-Norm:** 정규화 위치 최적화
- **RMSNorm:** Layer Norm의 효율적 대안
- **SwiGLU:** 향상된 활성화 함수

### 7.3. 위치 인코딩 발전
- **RoPE:** 회전 위치 임베딩
- **ALiBi:** 어텐션 편향을 통한 위치 정보
- **xPos:** 외삽 성능 개선된 위치 인코딩

이러한 구성 요소들의 결합을 통해 트랜스포머는 현대 AI의 핵심 아키텍처로 자리 잡았으며, 지속적인 발전을 통해 다양한 도메인에서 혁신적인 성능을 달성하고 있습니다.
