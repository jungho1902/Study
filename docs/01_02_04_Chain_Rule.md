# 연쇄 법칙 (Chain Rule)

연쇄 법칙은 **합성 함수(composite function)**를 미분하는 방법을 제공하는 강력한 규칙입니다. 합성 함수란 여러 함수가 '연쇄적으로' 연결된 형태, 즉 한 함수의 출력이 다른 함수의 입력으로 사용되는 함수를 말합니다 (예: $`f(g(x))`$).

연쇄 법칙은 딥러닝에서 **역전파(Backpropagation)** 알고리즘의 핵심적인 수학 원리입니다. 출력층의 손실(error)이 입력층 방향으로 각 층을 거슬러 전파될 때, 각 파라미터가 손실에 얼마나 영향을 미쳤는지(그래디언트) 계산하기 위해 연쇄 법칙이 사용됩니다.

---

### 1. 단일 변수 연쇄 법칙 (Chain Rule for Single Variable)

함수 $`y = f(u)`$와 $`u = g(x)`$가 각각 미분 가능할 때, 이들의 합성 함수 $`y = f(g(x))`$의 도함수는 다음과 같습니다.

$`\frac{dy}{dx} = \frac{dy}{du} \cdot \frac{du}{dx}`$

- **해석:** $`y`$의 $`x`$에 대한 변화율은, ($`y`$의 $`u`$에 대한 변화율)과 ($`u`$의 $`x`$에 대한 변화율)의 곱과 같습니다. 즉, '바깥 함수'를 미분하고, 그 결과에 '안쪽 함수'의 미분을 곱하는 것입니다.

---

### 2. 다변수 연쇄 법칙 (Multivariable Chain Rule)

머신러닝에서는 여러 변수가 복잡하게 얽힌 다변수 함수의 연쇄 법칙이 더 중요합니다.

예를 들어, 변수 $`x, y`$가 중간 변수 $`u, v`$를 통해 최종적으로 $`z`$에 영향을 미친다고 가정해 봅시다.
- $`z = f(u, v)`$
- $`u = g(x, y)`$
- $`v = h(x, y)`$

이때, 최종 출력 $`z`$의 입력 변수 $`x`$에 대한 편미분($`\frac{\partial z}{\partial x}`$)은 $`x`$가 $`z`$에 영향을 미치는 **모든 경로**를 고려하여 계산해야 합니다.
- 경로 1: $`x \to u \to z`$
- 경로 2: $`x \to v \to z`$

각 경로의 변화율을 연쇄 법칙으로 구하고, 이들을 모두 더해줍니다.

$`\frac{\partial z}{\partial x} = \frac{\partial z}{\partial u}\frac{\partial u}{\partial x} + \frac{\partial z}{\partial v}\frac{\partial v}{\partial x}`$

마찬가지로, $`y`$에 대한 편미분은 다음과 같습니다.
$`\frac{\partial z}{\partial y} = \frac{\partial z}{\partial u}\frac{\partial u}{\partial y} + \frac{\partial z}{\partial v}\frac{\partial v}{\partial y}`$

이 원리는 신경망의 역전파에서 출력층의 손실로부터 각 가중치($`w_{ij}`$)에 대한 그래디언트를 계산할 때 그대로 적용됩니다.

---

## 예제 및 풀이 (Examples and Solutions)

### 예제 1: 단일 변수 연쇄 법칙

**문제:** 함수 $`y = (x^2 + 1)^3`$의 도함수 $`\frac{dy}{dx}`$를 구하시오.

**풀이:**
이 함수를 두 개의 함수로 분리할 수 있습니다.
- $`y = u^3`$ (바깥 함수)
- $`u = x^2 + 1`$ (안쪽 함수)

연쇄 법칙 $`\frac{dy}{dx} = \frac{dy}{du} \cdot \frac{du}{dx}`$를 적용합니다.

1.  **$`\frac{dy}{du}`$ 계산:**
    - $`y = u^3`$을 $`u`$에 대해 미분합니다.
    - $`\frac{dy}{du} = 3u^2`$

2.  **$`\frac{du}{dx}`$ 계산:**
    - $`u = x^2 + 1`$을 $`x`$에 대해 미분합니다.
    - $`\frac{du}{dx} = 2x`$

3.  **결과 조합:**
    - $`\frac{dy}{dx} = (3u^2) \cdot (2x)`$
    - 마지막으로, $`u`$를 다시 원래의 $`x`$에 대한 식으로 바꿔줍니다 ($`u=x^2+1`$).
    - $`\frac{dy}{dx} = 3(x^2 + 1)^2 \cdot (2x) = 6x(x^2 + 1)^2`$

**답:** $`\frac{dy}{dx} = 6x(x^2 + 1)^2`$

### 예제 2: 다변수 연쇄 법칙

**문제:** $`z = u^2 + v^3`$ 이고, $`u = 2x+y`$, $`v = x-y`$ 일 때, $`\frac{\partial z}{\partial x}`$를 구하시오.

**풀이:**
$`x`$가 $`z`$에 영향을 미치는 두 경로($`x \to u \to z`$, $`x \to v \to z`$)를 모두 고려해야 합니다.
공식: $`\frac{\partial z}{\partial x} = \frac{\partial z}{\partial u}\frac{\partial u}{\partial x} + \frac{\partial z}{\partial v}\frac{\partial v}{\partial x}`$

1.  **각 편미분 계산:**
    - $`\frac{\partial z}{\partial u} = 2u`$
    - $`\frac{\partial z}{\partial v} = 3v^2`$
    - $`\frac{\partial u}{\partial x} = 2`$
    - $`\frac{\partial v}{\partial x} = 1`$

2.  **공식에 대입:**
    - $`\frac{\partial z}{\partial x} = (2u) \cdot (2) + (3v^2) \cdot (1) = 4u + 3v^2`$

3.  **$`u, v`$를 $`x, y`$에 대한 식으로 변환:**
    - $`\frac{\partial z}{\partial x} = 4(2x+y) + 3(x-y)^2`$
    - $`= 8x + 4y + 3(x^2 - 2xy + y^2)`$
    - $`= 8x + 4y + 3x^2 - 6xy + 3y^2`$

**답:** $`\frac{\partial z}{\partial x} = 3x^2 - 6xy + 3y^2 + 8x + 4y`$
