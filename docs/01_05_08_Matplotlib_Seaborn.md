# Matplotlib/Seaborn: 데이터 시각화

**데이터 시각화(Data Visualization)**는 데이터를 그래프나 차트 등 시각적인 형태로 표현하여, 데이터에 숨겨진 패턴, 추세, 관계를 쉽게 파악할 수 있도록 하는 과정입니다. 파이썬에서는 **Matplotlib**와 **Seaborn**이 가장 대표적인 데이터 시각화 라이브러리입니다.

---

### 1. Matplotlib

**Matplotlib**는 파이썬에서 가장 기본적이고 널리 사용되는 시각화 라이브러리입니다. 다양한 종류의 정적(static), 애니메이션, 인터랙티브한 시각화를 만들 수 있는 강력한 기능을 제공합니다. Matplotlib는 다른 많은 시각화 라이브러리들의 기반이 됩니다.

- `pyplot` 모듈을 `plt`라는 별칭으로 가져와 사용하는 것이 관례입니다. (`import matplotlib.pyplot as plt`)

#### 기본 플롯(Plot) 생성

Matplotlib 시각화는 보통 다음의 절차를 따릅니다.
1. `plt.figure()`로 그림을 그릴 캔버스(Figure)를 준비합니다.
2. `plt.plot()`과 같은 함수로 원하는 그래프를 그립니다.
3. `plt.title()`, `plt.xlabel()`, `plt.ylabel()` 등으로 제목과 축 라벨을 추가합니다.
4. `plt.show()`로 최종 결과물을 화면에 표시합니다.

```python
import matplotlib.pyplot as plt

# 데이터 준비
x = [1, 2, 3, 4, 5]
y = [2, 3, 5, 7, 11]

# 1. Figure 생성
plt.figure(figsize=(8, 5)) # 캔버스 크기 조절

# 2. 라인 플롯 그리기
plt.plot(x, y)

# 3. 제목 및 축 라벨 추가
plt.title("Simple Line Plot")
plt.xlabel("X-axis")
plt.ylabel("Y-axis")

# 4. 화면에 표시
plt.show()
```

---

### 2. Seaborn

**Seaborn**은 Matplotlib을 기반으로 만들어진 라이브러리로, 더 아름답고 통계적으로 의미 있는 시각화를 더 쉽게 만들 수 있도록 설계되었습니다. 특히 Pandas DataFrame과 매우 잘 통합되어 작동합니다.

- `import seaborn as sns`로 라이브러리를 가져오는 것이 관례입니다.

#### 통계적 플롯 생성

Seaborn을 사용하면 복잡한 통계적 그래프(예: 분포도, 관계도)를 단 몇 줄의 코드로 그릴 수 있습니다.

```python
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

# Pandas DataFrame 준비
data = {'temperature': [25, 28, 22, 30, 31, 29, 26],
        'sales': [50, 60, 45, 70, 72, 65, 55]}
df = pd.DataFrame(data)

# Seaborn을 이용한 산점도(Scatter Plot) 그리기
plt.figure(figsize=(8, 5))
sns.scatterplot(x='temperature', y='sales', data=df)

# 제목 및 축 라벨 추가
plt.title("Temperature vs. Sales")
plt.xlabel("Temperature (°C)")
plt.ylabel("Sales (units)")

# 화면에 표시
plt.show()
```

---

### 3. 다양한 종류의 그래프

Matplotlib와 Seaborn을 사용하면 다음과 같은 다양한 그래프를 손쉽게 생성할 수 있습니다.

- **라인 플롯 (Line Plot):** 시간의 흐름에 따른 데이터 변화를 보여주는 데 적합합니다. (`plt.plot`, `sns.lineplot`)
- **산점도 (Scatter Plot):** 두 변수 간의 관계를 파악하는 데 사용됩니다. (`plt.scatter`, `sns.scatterplot`)
- **막대 그래프 (Bar Plot):** 범주형 데이터의 크기를 비교하는 데 사용됩니다. (`plt.bar`, `sns.barplot`)
- **히스토그램 (Histogram):** 연속형 데이터의 분포를 시각화합니다. (`plt.hist`, `sns.histplot`)
- **박스 플롯 (Box Plot):** 데이터의 사분위수, 중앙값, 이상치 등을 시각적으로 표현합니다. (`plt.box`, `sns.boxplot`)
- **히트맵 (Heatmap):** 행렬 형태의 데이터 값을 색상으로 표현하여 패턴을 파악하는 데 유용합니다. (`plt.imshow`, `sns.heatmap`)

데이터의 종류와 분석의 목적에 맞는 적절한 시각화 방법을 선택하는 것이 효과적인 데이터 분석의 핵심입니다.
