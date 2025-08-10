# 제어문 (조건문, 반복문)

제어문은 프로그램의 실행 흐름을 제어하는 데 사용됩니다. 특정 조건에 따라 다른 코드를 실행하게 하거나, 특정 코드를 여러 번 반복하게 만들 수 있습니다. 파이썬의 주요 제어문에는 조건문(`if`)과 반복문(`for`, `while`)이 있습니다.

---

### 1. 조건문 (Conditional Statements)

조건문은 주어진 조건이 참(True)인지 거짓(False)인지에 따라 다른 코드 블록을 실행합니다.

#### 가. `if`
- 가장 기본적인 조건문으로, `if` 뒤의 조건이 `True`일 경우에만 내부의 코드 블록을 실행합니다.
- 코드 블록은 들여쓰기(indentation)로 구분됩니다. (보통 공백 4칸)

```python
score = 95
if score > 90:
    print("축하합니다! 합격입니다.")
    print("성적이 매우 우수합니다.")
```

#### 나. `if-else`
- `if` 조건이 `False`일 경우, `else` 블록의 코드를 실행합니다.

```python
score = 85
if score > 90:
    print("합격입니다.")
else:
    print("아쉽지만 불합격입니다.")
```

#### 다. `if-elif-else`
- 여러 개의 조건을 순차적으로 확인할 때 사용합니다.
- `elif`는 'else if'의 줄임말로, 여러 개를 사용할 수 있습니다.

```python
score = 85
if score >= 90:
    print("A 등급")
elif score >= 80:
    print("B 등급") # 이 조건이 참이므로 이 블록이 실행되고, 조건문은 종료됩니다.
elif score >= 70:
    print("C 등급")
else:
    print("F 등급")
```

---

### 2. 반복문 (Loops)

반복문은 특정 코드 블록을 여러 번 실행할 때 사용됩니다.

#### 가. `for` 반복문
- 리스트, 튜플, 문자열 등 순회 가능한(iterable) 객체의 각 요소를 하나씩 순회하며 코드 블록을 실행합니다.
- `range()` 함수와 함께 사용하면 특정 횟수만큼 반복하는 데 유용합니다.

```python
# 리스트 순회
fruits = ["apple", "banana", "cherry"]
for fruit in fruits:
    print(fruit)

# range() 함수 사용
# range(5)는 0, 1, 2, 3, 4를 의미합니다.
for i in range(5):
    print(f"현재 숫자는 {i}입니다.")
```

#### 나. `while` 반복문
- 주어진 조건이 `True`인 동안 코드 블록을 계속해서 반복합니다.
- 조건이 `False`가 되면 반복을 멈춥니다.
- 반복을 멈추기 위한 조건 변경 코드가 루프 내에 반드시 있어야 무한 루프에 빠지지 않습니다.

```python
count = 0
while count < 5:
    print(f"현재 숫자는 {count}입니다.")
    count = count + 1 # count 값을 1씩 증가시켜 언젠가 조건이 False가 되게 함
```

#### 다. `break`와 `continue`
- `break`: 반복문 실행 중 `break`를 만나면, 즉시 해당 반복문을 완전히 빠져나옵니다.
- `continue`: 반복문 실행 중 `continue`를 만나면, 현재 반복의 나머지 부분을 건너뛰고 바로 다음 반복을 시작합니다.

```python
# break 예시: 5를 만나면 반복 중단
for i in range(10):
    if i == 5:
        break
    print(i) # 0, 1, 2, 3, 4 까지만 출력됨

# continue 예시: 짝수일 경우 건너뛰기
for i in range(10):
    if i % 2 == 0:
        continue
    print(i) # 1, 3, 5, 7, 9 만 출력됨
```
