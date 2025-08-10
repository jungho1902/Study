# 행렬식 (Determinant)

행렬식은 **정사각행렬**에 대해서만 정의되는 특별한 스칼라 값입니다. 기호로는 $`\det(A)`$ 또는 $`|A|`$로 표기합니다. 행렬식은 행렬이 나타내는 선형 변환(linear transformation)의 기하학적 특성을 설명하는 중요한 지표입니다.

---

### 1. 행렬식의 의미 (Geometric Interpretation)

행렬식의 절대값은 해당 행렬이 만드는 선형 변환에 의해 공간이 얼마나 확대 또는 축소되는지를 나타내는 **'스케일링 팩터(scaling factor)'**입니다.

- **2x2 행렬:** 행렬의 두 열벡터가 만드는 **평행사변형의 넓이**를 나타냅니다.
- **3x3 행렬:** 행렬의 세 열벡터가 만드는 **평행육면체(parallelepiped)의 부피**를 나타냅니다.
- **$|A| = 1$**: 변환 시 넓이나 부피가 보존됩니다.
- **$|A| > 1$**: 변환 시 넓이나 부피가 확대됩니다.
- **$|A| < 1$**: 변환 시 넓이나 부피가 축소됩니다.
- **$|A| = 0$**: 변환 시 차원이 축소됩니다 (예: 2D 평면이 1D 선으로 찌그러짐). 이는 행렬의 열벡터들이 선형 종속(linearly dependent)임을 의미하며, **역행렬이 존재하지 않음**을 시사합니다.
- **$|A| < 0$**: 변환 시 공간의 방향(orientation)이 뒤집힙니다 (예: 거울상 변환).

---

### 2. 행렬식의 계산 (Calculation)

#### 가. 2x2 행렬
행렬 $`A = \begin{bmatrix} a & b \\ c & d \end{bmatrix}`$의 행렬식은 다음과 같습니다.
$`\det(A) = ad - bc`$

#### 나. 3x3 행렬
행렬 $`A = \begin{bmatrix} a & b & c \\ d & e & f \\ g & h & i \end{bmatrix}`$의 행렬식은 **사뤼스(Sarrus)의 법칙**이나 **여인수 전개(Cofactor Expansion)**를 통해 계산할 수 있습니다. 여기서는 여인수 전개를 사용합니다.
$`\det(A) = a \begin{vmatrix} e & f \\ h & i \end{vmatrix} - b \begin{vmatrix} d & f \\ g & i \end{vmatrix} + c \begin{vmatrix} d & e \\ g & h \end{vmatrix}`$
$`= a(ei - fh) - b(di - fg) + c(dh - eg)`$

---

## 예제 및 풀이 (Examples and Solutions)

### 예제 1: 2x2 행렬의 행렬식

**문제:** 행렬 $`A = \begin{bmatrix} 3 & -1 \\ 2 & 5 \end{bmatrix}`$의 행렬식을 계산하시오.

**풀이:**
공식 $`\det(A) = ad - bc`$를 사용합니다.
- $`a=3, b=-1, c=2, d=5`$
- $`\det(A) = (3 \times 5) - (-1 \times 2) = 15 - (-2) = 17`$

**답:** $`\det(A) = 17`$

### 예제 2: 3x3 행렬의 행렬식

**문제:** 행렬 $`B = \begin{bmatrix} 1 & 2 & 3 \\ 0 & 4 & 5 \\ 1 & 0 & 6 \end{bmatrix}`$의 행렬식을 계산하시오.

**풀이:**
첫 번째 행을 기준으로 여인수 전개를 사용합니다.
$`\det(B) = 1 \begin{vmatrix} 4 & 5 \\ 0 & 6 \end{vmatrix} - 2 \begin{vmatrix} 0 & 5 \\ 1 & 6 \end{vmatrix} + 3 \begin{vmatrix} 0 & 4 \\ 1 & 0 \end{vmatrix}`$

1.  **각 2x2 부분 행렬식 계산:**
    - $`\begin{vmatrix} 4 & 5 \\ 0 & 6 \end{vmatrix} = (4 \times 6) - (5 \times 0) = 24`$
    - $`\begin{vmatrix} 0 & 5 \\ 1 & 6 \end{vmatrix} = (0 \times 6) - (5 \times 1) = -5`$
    - $`\begin{vmatrix} 0 & 4 \\ 1 & 0 \end{vmatrix} = (0 \times 0) - (4 \times 1) = -4`$

2.  **결과를 조합:**
    - $`\det(B) = 1(24) - 2(-5) + 3(-4) = 24 + 10 - 12 = 22`$

**답:** $`\det(B) = 22`$
