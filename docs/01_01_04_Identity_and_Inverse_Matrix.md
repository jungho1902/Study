# 단위행렬(Identity Matrix)과 역행렬(Inverse Matrix)

단위행렬과 역행렬은 선형대수에서 곱셈 연산에 대한 항등원과 역원을 나타내는 중요한 개념입니다. 이는 방정식을 풀고, 데이터를 변환하는 등 다양한 응용의 기초가 됩니다.

---

### 1. 단위행렬 (Identity Matrix)

단위행렬은 주대각선(main diagonal)의 원소가 모두 1이고 나머지 원소는 모두 0인 정사각행렬(square matrix)을 말하며, 기호는 **$`I`$** 또는 $`I_n`$ (크기를 명시할 때)으로 표기합니다.

**정의:**
단위행렬은 행렬 곱셈의 항등원(identity element)입니다. 즉, 어떤 행렬 $`A`$에 단위행렬 $`I`$를 곱해도 행렬 $`A`$는 변하지 않습니다.

- $`AI = IA = A`$

**형태:**
- $`I_2 = \begin{bmatrix} 1 & 0 \\ 0 & 1 \end{bmatrix}`$
- $`I_3 = \begin{bmatrix} 1 & 0 & 0 \\ 0 & 1 & 0 \\ 0 & 0 & 1 \end{bmatrix}`$

**의미:**
단위행렬은 기하학적으로 '아무 변환도 하지 않음'을 의미합니다. 어떤 벡터에 단위행렬을 곱해도 해당 벡터는 원래의 위치와 방향을 그대로 유지합니다.

---

### 2. 역행렬 (Inverse Matrix)

역행렬은 행렬 곱셈에 대한 역원(inverse element)의 개념입니다. 어떤 행렬 $`A`$와 그 역행렬 $`A^{-1}`$를 곱하면 단위행렬 $`I`$가 됩니다.

**조건:**
- **정사각행렬이어야만 합니다.**
- **행렬식(Determinant)이 0이 아니어야 합니다.** 행렬식이 0인 행렬은 **특이 행렬(Singular Matrix)** 또는 비가역 행렬(non-invertible matrix)이라고 하며, 역행렬이 존재하지 않습니다.

**정의:**
정사각행렬 $`A`$에 대하여 다음을 만족하는 행렬 $`A^{-1}`$를 $`A`$의 역행렬이라고 합니다.

- $`AA^{-1} = A^{-1}A = I`$

**의미:**
역행렬은 어떤 선형 변환을 '취소'하는 변환을 의미합니다. 예를 들어, 벡터 $`\mathbf{v}`$를 행렬 $`A`$로 변환하여 $`\mathbf{w} = A\mathbf{v}`$를 얻었다면, 변환된 벡터 $`\mathbf{w}`$에 $`A^{-1}`$를 곱하여 원래 벡터 $`\mathbf{v}`$로 되돌릴 수 있습니다. ($`A^{-1}\mathbf{w} = A^{-1}(A\mathbf{v}) = (A^{-1}A)\mathbf{v} = I\mathbf{v} = \mathbf{v}`$)

---

## 예제 및 풀이 (Examples and Solutions)

### 예제 1: 2x2 행렬의 역행렬 계산

**문제:** 행렬 $`A = \begin{bmatrix} 4 & 7 \\ 2 & 6 \end{bmatrix}`$ 의 역행렬 $`A^{-1}`$를 구하시오.

**풀이:**
2x2 행렬 $`A = \begin{bmatrix} a & b \\ c & d \end{bmatrix}`$ 의 역행렬은 다음 공식을 통해 구할 수 있습니다.

$`A^{-1} = \frac{1}{ad-bc} \begin{bmatrix} d & -b \\ -c & a \end{bmatrix}`$

여기서 $`ad-bc`$는 행렬 $`A`$의 행렬식(determinant)입니다.

1.  **행렬식 계산:**
    - $`\det(A) = ad - bc = (4 \times 6) - (7 \times 2) = 24 - 14 = 10`$
    - 행렬식이 0이 아니므로 역행렬이 존재합니다.

2.  **공식에 대입:**
    - 주대각선 원소($`a, d`$)의 위치를 바꿉니다: $`\begin{bmatrix} 6 & 7 \\ 2 & 4 \end{bmatrix}`$
    - 나머지 원소($`b, c`$)의 부호를 바꿉니다: $`\begin{bmatrix} 6 & -7 \\ -2 & 4 \end{bmatrix}`$
    - 행렬식의 역수를 곱합니다: $`A^{-1} = \frac{1}{10} \begin{bmatrix} 6 & -7 \\ -2 & 4 \end{bmatrix}`$

**답:**
$`A^{-1} = \begin{bmatrix} 0.6 & -0.7 \\ -0.2 & 0.4 \end{bmatrix}`$

**확인:** $`AA^{-1} = I`$ 인지 확인해 봅니다.
$`\begin{bmatrix} 4 & 7 \\ 2 & 6 \end{bmatrix} \begin{bmatrix} 0.6 & -0.7 \\ -0.2 & 0.4 \end{bmatrix} = \begin{bmatrix} (4 \times 0.6 + 7 \times -0.2) & (4 \times -0.7 + 7 \times 0.4) \\ (2 \times 0.6 + 6 \times -0.2) & (2 \times -0.7 + 6 \times 0.4) \end{bmatrix}`$
$`= \begin{bmatrix} (2.4 - 1.4) & (-2.8 + 2.8) \\ (1.2 - 1.2) & (-1.4 + 2.4) \end{bmatrix} = \begin{bmatrix} 1 & 0 \\ 0 & 1 \end{bmatrix} = I`$
결과가 단위행렬이므로 계산이 정확합니다.
