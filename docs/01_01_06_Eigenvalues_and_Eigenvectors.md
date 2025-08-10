# 고유값(Eigenvalues)과 고유벡터(Eigenvectors)

고유값과 고유벡터는 선형대수에서 가장 중요한 개념 중 하나로, 행렬이 나타내는 선형 변환의 핵심적인 구조를 설명합니다. 이들은 행렬을 '분해'하여 그 동작을 더 쉽게 이해하도록 돕고, 주성분 분석(PCA)과 같은 여러 머신러닝 알고리즘의 근간을 이룹니다.

---

### 1. 고유값과 고유벡터의 정의

정사각행렬 $`A`$에 대하여, 아래의 방정식을 만족하는 '0이 아닌' 벡터 $`\mathbf{v}`$와 스칼라 $`\lambda`$가 존재할 때,

$`A\mathbf{v} = \lambda\mathbf{v}`$

- $`\lambda`$ (람다)를 행렬 $`A`$의 **고유값(Eigenvalue)**이라고 합니다.
- $`\mathbf{v}`$를 그 고유값 $`\lambda`$에 해당하는 **고유벡터(Eigenvector)**라고 합니다.

**기하학적 의미:**
고유벡터는 행렬 $`A`$가 나타내는 선형 변환을 가했을 때, **방향이 변하지 않고 크기만 변하는** 특별한 벡터입니다. 고유값은 그 고유벡터의 크기가 변하는 **스케일링 팩터(scaling factor)**입니다.
- $`\lambda > 1`$: 고유벡터의 방향으로 늘어납니다.
- $`0 < \lambda < 1`$: 고유벡터의 방향으로 줄어듭니다.
- $`\lambda < 0`$: 방향이 반대가 되며 크기가 조절됩니다.

---

### 2. 고유값과 고유벡터의 계산

위의 정의 식을 변형하여 계산 방법을 유도할 수 있습니다.
$`A\mathbf{v} - \lambda\mathbf{v} = 0`$
$`A\mathbf{v} - \lambda I \mathbf{v} = 0`$
$`(A - \lambda I)\mathbf{v} = 0`$

이 방정식을 만족하는 '0이 아닌' 벡터 $`\mathbf{v}`$가 존재하려면, 행렬 $`(A - \lambda I)`$는 비가역적(non-invertible)이어야 합니다. 즉, 이 행렬의 행렬식(determinant)이 0이어야 합니다.

- **특성방정식(Characteristic Equation):** $`\det(A - \lambda I) = 0`$

**계산 절차:**
1.  **고유값 찾기:** 특성방정식 $`\det(A - \lambda I) = 0`$을 풀어 스칼라 값 $`\lambda`$를 찾습니다.
2.  **고유벡터 찾기:** 각 고유값 $`\lambda`$를 다시 $`(A - \lambda I)\mathbf{v} = 0`$에 대입하여, 이 방정식을 만족하는 벡터 $`\mathbf{v}`$를 찾습니다.

---

## 예제 및 풀이 (Examples and Solutions)

### 예제 1: 2x2 행렬의 고유값과 고유벡터

**문제:** 행렬 $`A = \begin{bmatrix} 2 & 1 \\ 1 & 2 \end{bmatrix}`$의 고유값과 그에 해당하는 고유벡터를 찾으시오.

**풀이:**

**1. 고유값 찾기**
특성방정식 $`\det(A - \lambda I) = 0`$을 이용합니다.
$`A - \lambda I = \begin{bmatrix} 2 & 1 \\ 1 & 2 \end{bmatrix} - \lambda \begin{bmatrix} 1 & 0 \\ 0 & 1 \end{bmatrix} = \begin{bmatrix} 2-\lambda & 1 \\ 1 & 2-\lambda \end{bmatrix}`$

이제 행렬식을 계산합니다.
$`\det(A - \lambda I) = (2-\lambda)(2-\lambda) - (1)(1) = 0`$
$`\lambda^2 - 4\lambda + 4 - 1 = 0`$
$`\lambda^2 - 4\lambda + 3 = 0`$
$`(\lambda - 3)(\lambda - 1) = 0`$

따라서 고유값은 $`\lambda_1 = 3`$, $`\lambda_2 = 1`$ 입니다.

**2. 고유벡터 찾기**

**가. $`\lambda_1 = 3`$에 대한 고유벡터 $`\mathbf{v}_1`$**
$`(A - 3I)\mathbf{v}_1 = 0`$
$`\begin{bmatrix} 2-3 & 1 \\ 1 & 2-3 \end{bmatrix} \begin{bmatrix} x_1 \\ y_1 \end{bmatrix} = \begin{bmatrix} 0 \\ 0 \end{bmatrix}`$
$`\begin{bmatrix} -1 & 1 \\ 1 & -1 \end{bmatrix} \begin{bmatrix} x_1 \\ y_1 \end{bmatrix} = \begin{bmatrix} 0 \\ 0 \end{bmatrix}`$
두 방정식 모두 $`-x_1 + y_1 = 0`$, 즉 $`x_1 = y_1`$을 나타냅니다.
따라서, $`\mathbf{v}_1`$은 $`x_1 = y_1`$를 만족하는 0이 아닌 모든 벡터입니다.
일반적으로 크기를 1로 정규화하거나 간단한 정수 표현을 사용합니다. 예를 들어 $`x_1=1`$로 두면 $`y_1=1`$ 입니다.
$`\mathbf{v}_1 = \begin{bmatrix} 1 \\ 1 \end{bmatrix}`$

**나. $`\lambda_2 = 1`$에 대한 고유벡터 $`\mathbf{v}_2`$**
$`(A - 1I)\mathbf{v}_2 = 0`$
$`\begin{bmatrix} 2-1 & 1 \\ 1 & 2-1 \end{bmatrix} \begin{bmatrix} x_2 \\ y_2 \end{bmatrix} = \begin{bmatrix} 0 \\ 0 \end{bmatrix}`$
$`\begin{bmatrix} 1 & 1 \\ 1 & 1 \end{bmatrix} \begin{bmatrix} x_2 \\ y_2 \end{bmatrix} = \begin{bmatrix} 0 \\ 0 \end{bmatrix}`$
두 방정식 모두 $`x_2 + y_2 = 0`$, 즉 $`x_2 = -y_2`$를 나타냅니다.
예를 들어 $`x_2=1`$로 두면 $`y_2=-1`$ 입니다.
$`\mathbf{v}_2 = \begin{bmatrix} 1 \\ -1 \end{bmatrix}`$

**답:**
- 고유값 $`\lambda_1 = 3`$, 해당하는 고유벡터 $`\mathbf{v}_1 = c_1 \begin{bmatrix} 1 \\ 1 \end{bmatrix}`$ ($`c_1 \neq 0`$)
- 고유값 $`\lambda_2 = 1`$, 해당하는 고유벡터 $`\mathbf{v}_2 = c_2 \begin{bmatrix} 1 \\ -1 \end{bmatrix}`$ ($`c_2 \neq 0`$)
(고유벡터는 스칼라배를 해도 여전히 고유벡터이므로, 상수 $c$를 곱한 형태로 표현하기도 합니다.)
