# 내적(Dot Product) 및 외적(Cross Product)

벡터의 곱셈 연산에는 내적(Dot Product)과 외적(Cross Product) 두 가지 주요 방식이 있습니다. 이 두 연산은 결과의 형태(스칼라 vs. 벡터)와 기하학적 의미가 전혀 다르므로, 명확히 구분하여 이해하는 것이 중요합니다.

---

### 1. 내적 (Dot Product)

내적은 **스칼라 곱(scalar product)**이라고도 불리며, 두 벡터로부터 하나의 스칼라 값을 계산하는 연산입니다.

**정의:**
두 n차원 벡터 $`\mathbf{a} = [a_1, a_2, ..., a_n]`$ 와 $`\mathbf{b} = [b_1, b_2, ..., b_n]`$ 에 대한 내적은 다음과 같이 정의됩니다.

- **대수적 정의:** $`\mathbf{a} \cdot \mathbf{b} = \sum_{i=1}^{n} a_i b_i = a_1 b_1 + a_2 b_2 + \cdots + a_n b_n`$
- **기하학적 정의:** $`\mathbf{a} \cdot \mathbf{b} = \|\mathbf{a}\| \|\mathbf{b}\| \cos(\theta)`$
  - $`\|\mathbf{a}\|`$와 $`\|\mathbf{b}\|`$는 각 벡터의 크기(Euclidean norm)입니다.
  - $`\theta`$는 두 벡터 사이의 각도입니다.

**주요 속성 및 의미:**
- **결과:** 스칼라 값입니다.
- **교환 법칙 성립:** $`\mathbf{a} \cdot \mathbf{b} = \mathbf{b} \cdot \mathbf{a}`$
- **기하학적 의미:** 한 벡터를 다른 벡터에 투영(projection)한 크기와 관련이 있습니다. 내적 값은 두 벡터가 얼마나 '같은 방향을 가리키는지'를 나타냅니다.
  - $`\cos(\theta) > 0`$ (같은 방향): 내적은 양수.
  - $`\cos(\theta) < 0`$ (반대 방향): 내적은 음수.
  - $`\cos(\theta) = 0`$ (직교): 내적은 0.
- **머신러닝 응용:** 코사인 유사도(Cosine Similarity) 계산, 신경망의 가중합(weighted sum) 계산 등에 널리 사용됩니다.

---

### 2. 외적 (Cross Product)

외적은 **벡터 곱(vector product)**이라고도 불리며, **3차원 공간에서만** 정의되는 연산입니다. 외적의 결과는 입력된 두 벡터에 모두 직교(orthogonal)하는 새로운 벡터입니다.

**정의:**
두 3차원 벡터 $`\mathbf{a} = [a_1, a_2, a_3]`$ 와 $`\mathbf{b} = [b_1, b_2, b_3]`$ 에 대한 외적은 다음과 같이 정의됩니다.

- **대수적 정의:** $`\mathbf{a} \times \mathbf{b} = \begin{bmatrix} a_2 b_3 - a_3 b_2 \\ a_3 b_1 - a_1 b_3 \\ a_1 b_2 - a_2 b_1 \end{bmatrix}`$

**주요 속성 및 의미:**
- **결과:** 벡터 값입니다.
- **교환 법칙 성립 안 함 (반교환 법칙):** $`\mathbf{a} \times \mathbf{b} = -(\mathbf{b} \times \mathbf{a})`$
- **기하학적 의미:**
  - **방향:** 결과 벡터 $`\mathbf{a} \times \mathbf{b}`$는 벡터 $`\mathbf{a}`$와 $`\mathbf{b}`$가 이루는 평면에 수직입니다. 방향은 오른손 법칙(right-hand rule)을 따릅니다.
  - **크기:** 결과 벡터의 크기 $`\|\mathbf{a} \times \mathbf{b}\|`$는 두 벡터 $`\mathbf{a}`$와 $`\mathbf{b}`$가 만드는 평행사변형의 면적과 같습니다. ($`\|\mathbf{a} \times \mathbf{b}\| = \|\mathbf{a}\| \|\mathbf{b}\| \sin(\theta)`$)
- **응용:** 3D 그래픽스, 물리학 등에서 법선 벡터(normal vector) 계산, 토크(torque) 계산 등에 사용됩니다.

---

## 예제 및 풀이 (Examples and Solutions)

### 예제 1: 외적 (Cross Product)

**문제:** 3차원 공간의 두 벡터 $`\mathbf{a} = \begin{bmatrix} 2 \\ 3 \\ 4 \end{bmatrix}`$ 와 $`\mathbf{b} = \begin{bmatrix} 5 \\ 6 \\ 7 \end{bmatrix}`$ 가 주어졌을 때, 외적 $`\mathbf{a} \times \mathbf{b}`$를 계산하시오.

**풀이:**
외적의 대수적 정의에 따라 각 성분을 계산합니다.
$`\mathbf{a} \times \mathbf{b} = \begin{bmatrix} a_2 b_3 - a_3 b_2 \\ a_3 b_1 - a_1 b_3 \\ a_1 b_2 - a_2 b_1 \end{bmatrix}`$

1.  **첫 번째 성분 (x):**
    - $`a_2 b_3 - a_3 b_2 = (3 \times 7) - (4 \times 6) = 21 - 24 = -3`$

2.  **두 번째 성분 (y):**
    - $`a_3 b_1 - a_1 b_3 = (4 \times 5) - (2 \times 7) = 20 - 14 = 6`$

3.  **세 번째 성분 (z):**
    - $`a_1 b_2 - a_2 b_1 = (2 \times 6) - (3 \times 5) = 12 - 15 = -3`$

**답:** $`\mathbf{a} \times \mathbf{b} = \begin{bmatrix} -3 \\ 6 \\ -3 \end{bmatrix}`$

**확인:** 결과 벡터 $`[-3, 6, -3]`$는 원래 벡터 $`[2, 3, 4]`$ 및 $`[5, 6, 7]`$과 각각 내적했을 때 0이 되므로, 서로 직교함을 확인할 수 있습니다.
- $`(-3 \times 2) + (6 \times 3) + (-3 \times 4) = -6 + 18 - 12 = 0`$
- $`(-3 \times 5) + (6 \times 6) + (-3 \times 7) = -15 + 36 - 21 = 0`$
