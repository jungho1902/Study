# 파이썬 기본 문법 (변수, 자료형, 연산자)

파이썬은 간결하고 읽기 쉬운 문법을 가진 프로그래밍 언어로, 데이터 과학과 머신러닝 분야에서 가장 널리 사용됩니다. 이번 챕터에서는 파이썬 프로그래밍의 가장 기본적인 구성 요소인 변수, 자료형, 연산자에 대해 알아봅니다.

---

### 1. 주석 (Comments)

- 코드를 설명하는 글로, 프로그램 실행에는 영향을 주지 않습니다.
- `#` 기호를 사용하여 한 줄 주석을 작성합니다.

```python
# 이 라인은 주석입니다. 프로그램에 의해 무시됩니다.
print("Hello, World!") # 코드 뒷부분에 설명을 추가할 수도 있습니다.
```

---

### 2. 변수 (Variables)

변수는 데이터를 저장하기 위한 '이름표'가 붙은 상자와 같습니다. 특정 값을 저장하고 나중에 다시 사용하기 위해 변수를 사용합니다.

- **할당 (Assignment):** 등호(`=`)를 사용하여 변수에 값을 할당합니다.
- **규칙:**
  - 변수 이름은 영문자, 숫자, 밑줄(`_`)로 구성될 수 있습니다.
  - 숫자로 시작할 수 없습니다.
  - 대소문자를 구분합니다 (`my_var`와 `My_Var`는 다른 변수).

```python
# 변수 할당
x = 10
message = "Hello, Python!"

# 변수 값 출력
print(x)
print(message)

# 변수 값 변경
x = 20
print(x)
```

---

### 3. 기본 자료형 (Data Types)

파이썬은 다양한 종류의 데이터를 다룰 수 있으며, 각 데이터 종류를 '자료형'이라고 합니다.

- **정수 (Integer):** `int`. 소수점이 없는 숫자입니다. (예: `10`, `-5`, `0`)
- **실수 (Float):** `float`. 소수점이 있는 숫자입니다. (예: `3.14`, `-0.01`)
- **문자열 (String):** `str`. 텍스트 데이터를 나타냅니다. 작은따옴표(`'`)나 큰따옴표(`"`)로 감쌉니다.
- **불리언 (Boolean):** `bool`. 참(True) 또는 거짓(False) 두 가지 값만 가집니다.

```python
my_integer = 100
my_float = 3.14
my_string = "Data Science"
my_boolean = True # 첫 글자는 반드시 대문자

print(type(my_integer)) # <class 'int'>
print(type(my_float)) # <class 'float'>
print(type(my_string)) # <class 'str'>
print(type(my_boolean)) # <class 'bool'>
```

---

### 4. 연산자 (Operators)

연산자는 값에 대한 특정 연산을 수행하도록 하는 기호입니다.

#### 가. 산술 연산자 (Arithmetic Operators)
- `+` (더하기), `-` (빼기), `*` (곱하기), `/` (나누기)
- `**` (거듭제곱): `2 ** 3` -> 8
- `//` (몫): `10 // 3` -> 3
- `%` (나머지): `10 % 3` -> 1

#### 나. 비교 연산자 (Comparison Operators)
- `==` (같다), `!=` (다르다)
- `>` (크다), `<` (작다)
- `>=` (크거나 같다), `<=` (작거나 같다)
- 결과는 항상 `True` 또는 `False` (불리언) 입니다.

#### 다. 논리 연산자 (Logical Operators)
- `and`: 두 조건이 모두 참일 때 `True`.
- `or`: 두 조건 중 하나라도 참일 때 `True`.
- `not`: 조건의 결과를 반대로 뒤집습니다 (`True` -> `False`).

```python
# 산술 연산
a = 10
b = 3
print(a + b) # 13
print(a / b) # 3.333...
print(a % b) # 1

# 비교 연산
print(a > b) # True
print(a == 10) # True

# 논리 연산
is_student = True
has_book = False
print(is_student and has_book) # False
print(is_student or has_book) # True
print(not is_student) # False
```
