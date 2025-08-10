# 행렬 연산 (Matrix Operations)

행렬은 선형대수의 기본 구성 요소로, 데이터를 구조화하고 변환하는 데 사용됩니다. 행렬 연산은 머신러닝 모델의 학습 및 추론 과정에서 필수적인 역할을 수행합니다. 본 챕터에서는 기본적인 행렬 연산인 덧셈, 뺄셈, 스칼라 곱셈, 그리고 전치에 대해 다룹니다.

---

### 1. 행렬의 덧셈과 뺄셈 (Addition and Subtraction)

**조건:** 두 행렬의 덧셈과 뺄셈은 **두 행렬의 크기(차원)가 동일할 때**에만 정의됩니다.

**정의:** 두 $`m \times n`$ 행렬 $`A`$와 $`B`$가 주어졌을 때, 두 행렬의 합 $`A+B`$와 차 $`A-B`$는 같은 위치에 있는 원소들끼리 더하거나 빼서 얻는 새로운 $`m \times n`$ 행렬입니다.

- **덧셈:** $`(A+B)_{ij} = A_{ij} + B_{ij}`$
- **뺄셈:** $`(A-B)_{ij} = A_{ij} - B_{ij}`$

### 2. 스칼라 곱셈 (Scalar Multiplication)

**정의:** 행렬 $`A`$와 스칼라 $`c`$가 주어졌을 때, 스칼라 곱 $`cA`$는 행렬 $`A`$의 모든 원소에 스칼라 $`c`$를 곱하여 얻는 새로운 행렬입니다. 결과 행렬의 크기는 원래 행렬 $`A`$와 동일합니다.

- $`(cA)_{ij} = c \times A_{ij}`$

### 3. 전치 행렬 (Transpose Matrix)

**정의:** $`m \times n`$ 행렬 $`A`$의 전치 행렬(transpose matrix)은 $`A^T`$로 표기하며, $`A`$의 행과 열을 서로 맞바꾼 $`n \times m`$ 행렬입니다. 즉, 원래 행렬의 $`i`$번째 행은 전치 행렬의 $`i`$번째 열이 됩니다.

- $`(A^T)_{ij} = A_{ji}`$

**속성:**
- $`(A^T)^T = A`$
- $`(A+B)^T = A^T + B^T`$
- $`(cA)^T = cA^T`$
- $`(AB)^T = B^T A^T`$ (순서가 바뀜에 유의)

---

## 예제 및 풀이 (Examples and Solutions)

### 예제 1: 행렬 덧셈 및 스칼라 곱셈

**문제:** 두 행렬 $`A = \begin{bmatrix} 1 & 2 \\ 3 & 4 \end{bmatrix}`$ 와 $`B = \begin{bmatrix} 5 & 6 \\ 7 & 8 \end{bmatrix}`$ 가 주어졌을 때, $`2A + B`$를 계산하시오.

**풀이:**

1.  **스칼라 곱셈 $`2A`$ 계산:**
    $`2A = 2 \times \begin{bmatrix} 1 & 2 \\ 3 & 4 \end{bmatrix} = \begin{bmatrix} 2 \times 1 & 2 \times 2 \\ 2 \times 3 & 2 \times 4 \end{bmatrix} = \begin{bmatrix} 2 & 4 \\ 6 & 8 \end{bmatrix}`$

2.  **행렬 덧셈 $`(2A) + B`$ 계산:**
    $`2A + B = \begin{bmatrix} 2 & 4 \\ 6 & 8 \end{bmatrix} + \begin{bmatrix} 5 & 6 \\ 7 & 8 \end{bmatrix} = \begin{bmatrix} 2+5 & 4+6 \\ 6+7 & 8+8 \end{bmatrix} = \begin{bmatrix} 7 & 10 \\ 13 & 16 \end{bmatrix}`$

**답:** $`2A + B = \begin{bmatrix} 7 & 10 \\ 13 & 16 \end{bmatrix}`$

### 예제 2: 전치 행렬

**문제:** 행렬 $`A = \begin{bmatrix} 1 & 2 & 3 \\ 4 & 5 & 6 \end{bmatrix}`$ 의 전치 행렬 $`A^T`$를 구하시오.

**풀이:**
원래 행렬 $`A`$는 2x3 크기의 행렬입니다. 전치 행렬 $`A^T`$는 3x2 크기의 행렬이 됩니다. $`A`$의 1행은 $`A^T`$의 1열이 되고, $`A`$의 2행은 $`A^T`$의 2열이 됩니다.

- $`A`$의 1행: `[1, 2, 3]` -> $`A^T`$의 1열
- $`A`$의 2행: `[4, 5, 6]` -> $`A^T`$의 2열

**답:** $`A^T = \begin{bmatrix} 1 & 4 \\ 2 & 5 \\ 3 & 6 \end{bmatrix}`$
