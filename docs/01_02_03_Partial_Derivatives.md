# 편미분 (Partial Derivatives)

머신러닝 모델의 손실 함수(Loss Function)는 보통 수많은 파라미터(가중치, 편향 등)를 입력으로 갖는 다변수 함수(multivariable function)입니다. **편미분**은 이처럼 여러 변수를 가진 함수에서, **하나의 특정 변수에 대해서만**의 변화율을 측정하는 방법입니다.

---

### 1. 편미분의 개념

다변수 함수 $`f(x, y, ...)`$에서 편미분은 우리가 관심을 갖는 변수 하나를 제외한 나머지 모든 변수들을 **상수(constant)로 취급**하고 미분하는 것입니다.

- 함수 $`f(x, y)`$를 변수 $`x`$에 대해 편미분하는 것은, $`y`$를 잠시 상수로 간주하고 $`x`$에 대해 미분하는 것을 의미합니다.
- 기호로는 $`\frac{\partial f}{\partial x}`$ 또는 $`f_x`$로 표기합니다.

**기하학적 의미:**
다변수 함수의 그래프는 3차원 이상의 공간에 그려지는 곡면(surface)입니다. $`x`$에 대한 편미분 $`\frac{\partial f}{\partial x}`$는 이 곡면을 $`y`$축에 평행한 방향으로 잘랐을 때 나타나는 단면 곡선의 특정 지점에서의 기울기를 의미합니다. 즉, 전체 공간이 아닌 특정 축 방향으로의 변화율만을 분리해서 보는 것입니다.

### 2. 편미분과 그래디언트 (Gradient)

각 변수에 대한 편미분을 모두 계산하여 벡터 형태로 모아놓은 것을 **그래디언트(Gradient)**라고 합니다.
- 함수 $`f(x, y)`$의 그래디언트는 $`\nabla f = \begin{bmatrix} \frac{\partial f}{\partial x} \\ \frac{\partial f}{\partial y} \end{bmatrix}`$ 입니다.

그래디언트는 함수 값이 가장 가파르게 증가하는 방향을 가리키는 벡터이며, 그 크기는 해당 방향으로의 변화율을 나타냅니다. 경사 하강법에서는 이 그래디언트의 **반대 방향**으로 파라미터를 업데이트하여 손실 함수를 최소화합니다.

---

## 예제 및 풀이 (Examples and Solutions)

### 예제 1: 다변수 함수의 편미분

**문제:** 다음 다변수 함수 $`f(x, y) = 3x^2y + 2x^3 + y^4`$에 대하여, $`\frac{\partial f}{\partial x}`$ 와 $`\frac{\partial f}{\partial y}`$를 각각 계산하시오.

**풀이:**

**가. $`x`$에 대한 편미분 ($`\frac{\partial f}{\partial x}`$)**
이 경우, 변수 $`y`$를 상수처럼 취급합니다.

1.  **$`3x^2y`$ 편미분:**
    - $`3y`$를 상수로 보고 $`x^2`$을 미분합니다.
    - $`\frac{\partial}{\partial x}(3x^2y) = 3y \cdot (2x) = 6xy`$

2.  **$`2x^3`$ 편미분:**
    - $`\frac{\partial}{\partial x}(2x^3) = 2 \cdot (3x^2) = 6x^2`$

3.  **$`y^4`$ 편미분:**
    - $`y^4`$은 $`x`$에 대해 전체가 상수이므로, 미분하면 0이 됩니다.
    - $`\frac{\partial}{\partial x}(y^4) = 0`$

4.  **결과 조합:**
    - $`\frac{\partial f}{\partial x} = 6xy + 6x^2 + 0`$

**나. $`y`$에 대한 편미분 ($`\frac{\partial f}{\partial y}`$)**
이 경우, 변수 $`x`$를 상수처럼 취급합니다.

1.  **$`3x^2y`$ 편미분:**
    - $`3x^2`$을 상수로 보고 $`y`$를 미분합니다. ($`y`$의 미분은 1)
    - $`\frac{\partial}{\partial y}(3x^2y) = 3x^2 \cdot (1) = 3x^2`$

2.  **$`2x^3`$ 편미분:**
    - $`2x^3`$은 $`y`$에 대해 전체가 상수이므로, 미분하면 0이 됩니다.
    - $`\frac{\partial}{\partial y}(2x^3) = 0`$

3.  **$`y^4`$ 편미분:**
    - $`\frac{\partial}{\partial y}(y^4) = 4y^3`$

4.  **결과 조합:**
    - $`\frac{\partial f}{\partial y} = 3x^2 + 0 + 4y^3`$

**답:**
- $`\frac{\partial f}{\partial x} = 6xy + 6x^2`$
- $`\frac{\partial f}{\partial y} = 3x^2 + 4y^3`$
