# 객체 지향 프로그래밍 (OOP) 기초

**객체 지향 프로그래밍(Object-Oriented Programming, OOP)**은 프로그램을 여러 개의 독립적인 **'객체(Object)'**들의 모임으로 보고, 이 객체들 간의 상호작용으로 서술하는 프로그래밍 패러다임입니다. OOP는 복잡한 문제를 더 작고 관리하기 쉬운 부분으로 나누어주어, 코드의 재사용성과 유지보수성을 높여줍니다.

---

### 1. 클래스(Class)와 객체(Object)

- **클래스 (Class):** 객체를 만들기 위한 '설계도' 또는 '틀'입니다. 클래스는 객체가 가질 속성(데이터)과 행동(기능)을 정의합니다.
  - 예: '자동차'의 설계도. 자동차는 '색상', '속도' 등의 속성을 가지고, '가속하기', '정지하기' 등의 행동을 할 수 있습니다.

- **객체 (Object):** 클래스로부터 만들어진 실제 '실체'입니다. 클래스라는 설계도를 바탕으로 메모리에 생성된 것을 **인스턴스(instance)**라고도 부릅니다.
  - 예: '빨간색 페라리', '파란색 트럭' 등은 '자동차' 클래스의 실제 객체(인스턴스)입니다.

---

### 2. 클래스 정의와 객체 생성

파이썬에서는 `class` 키워드를 사용하여 클래스를 정의합니다.

```python
# 'Dog' 라는 이름의 클래스를 정의
class Dog:
    # 클래스 변수 (모든 Dog 객체가 공유)
    species = "Canis lupus familiaris"

    # 생성자 (Initializer / Constructor)
    # 객체가 처음 생성될 때 호출되는 특별한 메서드
    def __init__(self, name, age):
        # 인스턴스 변수 (각 객체에 고유한 속성)
        self.name = name
        self.age = age
        print(f"{self.name}가 태어났습니다!")

    # 메서드 (Method)
    # 객체가 수행할 수 있는 행동
    def bark(self):
        print(f"{self.name}가 짖습니다: 멍멍!")

    def get_human_age(self):
        return self.age * 7

# 클래스로부터 객체(인스턴스)를 생성
my_dog = Dog("해피", 3)
your_dog = Dog("코코", 5)

# 객체의 속성에 접근
print(f"{my_dog.name}의 나이는 {my_dog.age}살입니다.") # 해피의 나이는 3살입니다.
print(f"{your_dog.name}의 나이는 {your_dog.age}살입니다.") # 코코의 나이는 5살입니다.

# 객체의 메서드 호출
my_dog.bark() # 해피가 짖습니다: 멍멍!
your_dog.bark() # 코코가 짖습니다: 멍멍!

# 메서드의 반환값 사용
human_equivalent_age = my_dog.get_human_age()
print(f"{my_dog.name}를 사람 나이로 환산하면 약 {human_equivalent_age}살입니다.")
```

---

### 3. OOP의 주요 구성요소

- **생성자 (`__init__`):** `__init__`는 객체가 생성될 때 파이썬에 의해 자동으로 호출되는 특별한 메서드입니다. `self`는 생성되는 객체 자기 자신을 가리키며, 이를 통해 각 객체의 고유한 속성(인스턴스 변수)을 초기화합니다.

- **`self`:** 클래스 내의 모든 메서드는 첫 번째 파라미터로 `self`를 가져야 합니다. `self`는 메서드를 호출한 객체 자신을 참조하며, 이를 통해 객체는 자신의 속성과 다른 메서드에 접근할 수 있습니다.

- **속성 (Attributes):** 객체의 데이터를 나타냅니다.
  - **클래스 변수:** 클래스의 모든 인스턴스가 공유하는 변수 (예: `Dog.species`).
  - **인스턴스 변수:** 각 인스턴스에 고유한 변수 (예: `my_dog.name`). `self.변수이름` 형태로 정의됩니다.

- **메서드 (Methods):** 객체의 행동을 나타내는 함수입니다. 클래스 내에 정의된 함수를 메서드라고 부릅니다.

OOP는 상속(Inheritance), 다형성(Polymorphism), 캡슐화(Encapsulation) 등 더 많은 개념을 포함하고 있으며, 이는 대규모 애플리케이션과 라이브러리를 구축하는 데 필수적입니다. 머신러닝 라이브러리인 `Scikit-learn`이나 딥러닝 프레임워크인 `PyTorch`, `TensorFlow`의 모델들도 모두 클래스 기반으로 구현되어 있습니다.
