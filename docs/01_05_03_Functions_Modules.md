# 함수와 모듈 (Functions and Modules)

함수와 모듈은 코드의 재사용성을 높이고, 프로그램을 구조적으로 관리할 수 있게 해주는 파이썬의 핵심적인 기능입니다.

---

### 1. 함수 (Functions)

**함수**는 특정 작업을 수행하는 코드 블록을 묶어서 이름을 붙인 것입니다. 함수를 사용하면 반복되는 코드를 여러 번 작성할 필요 없이, 필요할 때마다 함수의 이름을 호출하여 해당 코드를 실행할 수 있습니다.

#### 가. 함수 정의와 호출
- `def` 키워드를 사용하여 함수를 정의합니다.
- 함수 이름과 괄호`()` 안에 파라미터(parameter)를 지정할 수 있습니다.
- 코드 블록은 들여쓰기로 구분합니다.
- 정의된 함수는 이름을 호출하여 실행합니다.

```python
# 함수 정의
def greet(name):
    print(f"Hello, {name}!")

# 함수 호출
greet("Alice") # "Alice"는 인수(argument)
greet("Bob")
```

#### 나. 반환값 (Return Value)
- 함수는 `return` 키워드를 사용하여 실행 결과를 호출한 곳으로 되돌려줄 수 있습니다.
- `return`을 만나면 함수는 즉시 종료됩니다.

```python
def add(a, b):
    result = a + b
    return result

# 함수를 호출하고 반환값을 변수에 저장
sum_value = add(5, 3)
print(sum_value) # 8
```

#### 다. 기본값 파라미터 (Default Argument Values)
- 함수를 정의할 때 파라미터에 기본값을 지정할 수 있습니다.
- 함수 호출 시 해당 파라미터에 대한 인수가 전달되지 않으면, 지정된 기본값이 사용됩니다.

```python
def greet(name, message="안녕하세요"):
    print(f"{name}님, {message}.")

greet("김철수") # message 인수가 없으므로 기본값 "안녕하세요" 사용
greet("이영희", "반갑습니다") # message 인수가 있으므로 "반갑습니다" 사용
```

---

### 2. 모듈 (Modules)

**모듈**은 함수, 변수, 클래스 등을 모아놓은 파이썬 파일(`.py`)입니다. 모듈을 사용하면 다른 사람이 만든 유용한 기능들을 가져와서 사용하거나, 자신의 코드를 여러 파일로 나누어 체계적으로 관리할 수 있습니다.

#### 가. 모듈 가져오기 (`import`)
- `import` 키워드를 사용하여 다른 파이썬 파일을 현재 파일로 가져올 수 있습니다.
- 모듈 내의 함수나 변수를 사용할 때는 `모듈이름.함수이름` 형식으로 접근합니다.

```python
# math 모듈을 가져옴
import math

print(math.sqrt(16)) # 4.0 (제곱근 계산)
print(math.pi) # 3.141592... (원주율)

# as 키워드로 별칭(alias)을 지정할 수 있음
import math as m
print(m.sqrt(25)) # 5.0
```

#### 나. 특정 기능만 가져오기 (`from ... import ...`)
- 모듈 전체가 아닌, 필요한 특정 함수나 변수만 가져올 수 있습니다.
- 이 경우 `모듈이름.` 없이 바로 함수나 변수 이름을 사용할 수 있습니다.

```python
# math 모듈에서 sqrt 함수와 pi 변수만 가져옴
from math import sqrt, pi

print(sqrt(16)) # 4.0
print(pi) # 3.141592...

# 모듈의 모든 것을 가져오려면 *를 사용 (권장하지 않음)
# from math import *
```

**왜 모듈이 중요한가?**
데이터 과학과 머신러닝에서는 `NumPy`, `Pandas`, `Matplotlib`, `Scikit-learn`, `TensorFlow`, `PyTorch` 등 수많은 강력한 라이브러리들을 모듈 형태로 가져와 사용합니다. 모듈을 이해하는 것은 파이썬 생태계의 강력한 기능들을 활용하기 위한 첫걸음입니다.
