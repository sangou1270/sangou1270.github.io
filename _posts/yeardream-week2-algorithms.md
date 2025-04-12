---
layout: post
title: '이어드림스쿨 2주차: 알고리즘 기초'
date: 2024-04-25
categories: coding-education
tags: [yeardream-school, algorithms, big-o, sort, search]
toc: true
---

# 이어드림스쿨 2주차: 알고리즘 기초

## 알고리즘의 중요성

프로그래밍에서 알고리즘은 문제 해결의 핵심입니다. 알고리즘은 주어진 문제를 해결하기 위한 명확한 단계별 절차로, 효율적인 알고리즘을 설계하고 분석하는 능력은 모든 개발자에게 필수적인 역량입니다.

이어드림스쿨 2주차에서는 알고리즘의 기초부터 다양한 문제 해결 기법까지 배우며 실전 문제 해결 능력을 키웠습니다.

## 알고리즘 복잡도 분석

### 시간 복잡도 (Big-O Notation)

알고리즘의 성능을 평가하는 가장 일반적인 방법은 시간 복잡도를 분석하는 것입니다. Big-O 표기법은 알고리즘이 입력 크기에 따라 실행 시간이 어떻게 증가하는지를 나타냅니다.

주요 시간 복잡도:

- **O(1)** - 상수 시간: 입력 크기와 관계없이 일정한 시간이 소요됨
- **O(log n)** - 로그 시간: 이진 검색과 같은 알고리즘에서 발견됨
- **O(n)** - 선형 시간: 입력 크기에 비례하는 시간이 소요됨
- **O(n log n)** - 선형 로그 시간: 효율적인 정렬 알고리즘의 시간 복잡도
- **O(n²)** - 이차 시간: 중첩된 반복문에서 발견됨
- **O(2^n)** - 지수 시간: 조합 문제에서 발견되는 매우 비효율적인 복잡도

### 공간 복잡도

공간 복잡도는 알고리즘이 실행될 때 필요한 메모리 공간을 분석합니다. 시간과 공간은 종종 trade-off 관계에 있어, 문제 상황에 따라 적절한 균형을 찾아야 합니다.

## 기본 정렬 알고리즘

### 버블 정렬 (Bubble Sort)

버블 정렬은 가장 단순한 정렬 알고리즘 중 하나입니다. 인접한 두 요소를 비교하여 순서가 잘못되어 있으면 교환하는 방식으로 동작합니다.

```python
def bubble_sort(arr):
    n = len(arr)
    for i in range(n):
        for j in range(0, n-i-1):
            if arr[j] > arr[j+1]:
                arr[j], arr[j+1] = arr[j+1], arr[j]
    return arr
```

- 시간 복잡도: O(n²)
- 공간 복잡도: O(1)
- 안정적 정렬: Yes

### 선택 정렬 (Selection Sort)

선택 정렬은 배열에서 최솟값을 찾아 맨 앞으로 이동시키는 과정을 반복합니다.

```python
def selection_sort(arr):
    n = len(arr)
    for i in range(n):
        min_idx = i
        for j in range(i+1, n):
            if arr[j] < arr[min_idx]:
                min_idx = j
        arr[i], arr[min_idx] = arr[min_idx], arr[i]
    return arr
```

- 시간 복잡도: O(n²)
- 공간 복잡도: O(1)
- 안정적 정렬: No

### 삽입 정렬 (Insertion Sort)

삽입 정렬은 정렬된 부분과 정렬되지 않은 부분으로 배열을 나누고, 정렬되지 않은 요소를 정렬된 부분의 적절한 위치에 삽입합니다.

```python
def insertion_sort(arr):
    for i in range(1, len(arr)):
        key = arr[i]
        j = i - 1
        while j >= 0 and key < arr[j]:
            arr[j+1] = arr[j]
            j -= 1
        arr[j+1] = key
    return arr
```

- 시간 복잡도: 평균 O(n²), 최선의 경우 O(n)
- 공간 복잡도: O(1)
- 안정적 정렬: Yes

## 고급 정렬 알고리즘

### 병합 정렬 (Merge Sort)

병합 정렬은 분할 정복(divide and conquer) 전략을 사용하는 효율적인 정렬 알고리즘입니다. 배열을 절반으로 분할하고, 각 부분을 정렬한 다음 병합합니다.

```python
def merge_sort(arr):
    if len(arr) <= 1:
        return arr

    # 분할(Divide)
    mid = len(arr) // 2
    left = merge_sort(arr[:mid])
    right = merge_sort(arr[mid:])

    # 병합(Merge)
    return merge(left, right)

def merge(left, right):
    result = []
    i = j = 0

    while i < len(left) and j < len(right):
        if left[i] <= right[j]:
            result.append(left[i])
            i += 1
        else:
            result.append(right[j])
            j += 1

    result.extend(left[i:])
    result.extend(right[j:])
    return result
```

- 시간 복잡도: O(n log n)
- 공간 복잡도: O(n)
- 안정적 정렬: Yes

### 퀵 정렬 (Quick Sort)

퀵 정렬도 분할 정복 전략을 사용하지만, 피벗(pivot)을 기준으로 작은 값과 큰 값을 분할하는 방식으로 동작합니다.

```python
def quick_sort(arr):
    if len(arr) <= 1:
        return arr

    pivot = arr[len(arr) // 2]
    left = [x for x in arr if x < pivot]
    middle = [x for x in arr if x == pivot]
    right = [x for x in arr if x > pivot]

    return quick_sort(left) + middle + quick_sort(right)
```

- 시간 복잡도: 평균 O(n log n), 최악의 경우 O(n²)
- 공간 복잡도: O(log n) ~ O(n)
- 안정적 정렬: No (일반적인 구현)

## 검색 알고리즘

### 선형 검색 (Linear Search)

선형 검색은 배열의 모든 요소를 처음부터 순차적으로 확인하는 가장 단순한 검색 방법입니다.

```python
def linear_search(arr, target):
    for i in range(len(arr)):
        if arr[i] == target:
            return i  # 타겟 값의 인덱스 반환
    return -1  # 타겟 값이 없는 경우
```

- 시간 복잡도: O(n)
- 적용 상황: 정렬되지 않은 배열, 작은 데이터셋

### 이진 검색 (Binary Search)

이진 검색은 정렬된 배열에서 중간 값을 확인하고, 타겟이 중간 값보다 작으면 왼쪽 절반을, 크면 오른쪽 절반을 검색하는 방식으로 진행됩니다.

```python
def binary_search(arr, target):
    left, right = 0, len(arr) - 1

    while left <= right:
        mid = (left + right) // 2

        if arr[mid] == target:
            return mid  # 타겟 값의 인덱스 반환
        elif arr[mid] < target:
            left = mid + 1  # 오른쪽 절반 검색
        else:
            right = mid - 1  # 왼쪽 절반 검색

    return -1  # 타겟 값이 없는 경우
```

- 시간 복잡도: O(log n)
- 적용 상황: 정렬된 배열, 검색이 자주 필요한 대용량 데이터

## 재귀 알고리즘

재귀는 자기 자신을 호출하는 함수를 통해 문제를 해결하는 방법입니다. 복잡한 문제를 작은 하위 문제로 나누어 해결할 수 있습니다.

### 재귀의 기본 구조

```python
def recursive_function(parameters):
    # 기저 조건 (Base case)
    if termination_condition:
        return base_result

    # 재귀 호출
    return recursive_function(modified_parameters)
```

### 팩토리얼 예제

```python
def factorial(n):
    # 기저 조건
    if n == 0 or n == 1:
        return 1

    # 재귀 호출
    return n * factorial(n-1)
```

### 피보나치 수열 예제

```python
def fibonacci(n):
    # 기저 조건
    if n <= 1:
        return n

    # 재귀 호출
    return fibonacci(n-1) + fibonacci(n-2)
```

## 그리디 알고리즘

그리디(Greedy) 알고리즘은 각 단계에서 지역적으로 최적인 선택을 하는 전략입니다. 항상 전역적으로 최적인 해를 보장하지는 않지만, 특정 문제에서는 효율적인 해결책을 제공합니다.

### 동전 거스름돈 문제

```python
def coin_change_greedy(coins, amount):
    coins.sort(reverse=True)  # 큰 단위부터 처리
    result = 0

    for coin in coins:
        # 현재 동전으로 거슬러 줄 수 있는 최대 개수
        count = amount // coin
        result += count
        amount -= count * coin

    return result if amount == 0 else -1
```

### 활동 선택 문제

```python
def activity_selection(activities):
    # 종료 시간 기준으로 정렬
    activities.sort(key=lambda x: x[1])

    selected = [activities[0]]
    last_end_time = activities[0][1]

    for activity in activities[1:]:
        start_time, end_time = activity
        if start_time >= last_end_time:
            selected.append(activity)
            last_end_time = end_time

    return selected
```

## 다이나믹 프로그래밍 (DP)

다이나믹 프로그래밍은 복잡한 문제를 더 작은 하위 문제로 나누고, 하위 문제의 결과를 저장하여 중복 계산을 피하는 방법입니다.

### 메모이제이션 (Top-down)

```python
def fibonacci_memo(n, memo={}):
    if n in memo:
        return memo[n]

    if n <= 1:
        return n

    memo[n] = fibonacci_memo(n-1, memo) + fibonacci_memo(n-2, memo)
    return memo[n]
```

### 타뷸레이션 (Bottom-up)

```python
def fibonacci_tabulation(n):
    if n <= 1:
        return n

    dp = [0] * (n+1)
    dp[1] = 1

    for i in range(2, n+1):
        dp[i] = dp[i-1] + dp[i-2]

    return dp[n]
```

### 최장 증가 부분 수열 (LIS)

```python
def longest_increasing_subsequence(arr):
    n = len(arr)
    dp = [1] * n

    for i in range(1, n):
        for j in range(i):
            if arr[i] > arr[j]:
                dp[i] = max(dp[i], dp[j] + 1)

    return max(dp)
```

## 그래프 알고리즘

그래프는 노드와 간선으로 구성된 자료구조로, 다양한 관계와 네트워크를 표현할 수 있습니다.

### 깊이 우선 탐색 (DFS)

```python
def dfs(graph, start, visited=None):
    if visited is None:
        visited = set()

    visited.add(start)
    print(start, end=' ')

    for neighbor in graph[start]:
        if neighbor not in visited:
            dfs(graph, neighbor, visited)
```

### 너비 우선 탐색 (BFS)

```python
from collections import deque

def bfs(graph, start):
    visited = set([start])
    queue = deque([start])

    while queue:
        vertex = queue.popleft()
        print(vertex, end=' ')

        for neighbor in graph[vertex]:
            if neighbor not in visited:
                visited.add(neighbor)
                queue.append(neighbor)
```

## 실전 문제 해결 전략

### 문제 분석 프레임워크

1. **문제 이해**: 문제를 명확히 이해하고 요구사항을 파악합니다.
2. **접근 방법 설계**: 적절한 알고리즘과 자료구조를 선택합니다.
3. **알고리즘 구현**: 설계한 접근 방법을 코드로 구현합니다.
4. **테스트 및 최적화**: 예제 케이스로 테스트하고 필요시 최적화합니다.

### 효율적인 문제 해결을 위한 팁

- 시간 및 공간 복잡도를 항상 고려하세요.
- 극단적인 케이스(edge cases)를 고려하세요.
- 문제에 적합한 알고리즘과 자료구조를 선택하세요.
- 브루트 포스(brute force) 접근법부터 시작하여 점진적으로 최적화하세요.
- 비슷한 문제의 솔루션 패턴을 기억하고 적용하세요.

## 마무리

이어드림스쿨 2주차에서는 알고리즘의 기초부터 고급 개념까지 배우며 다양한 문제 해결 방법을 익혔습니다. 알고리즘 학습은 반복적인 연습이 중요하므로, 꾸준히 문제를 풀어보며 이론을 실전에 적용해 보세요.

다음 주차에서는 더 다양한 자료구조와 알고리즘 응용 사례를 배울 예정입니다.

## 추천 연습 문제

- [백준 온라인 저지](https://www.acmicpc.net/) - 기초 알고리즘 문제
- [프로그래머스](https://programmers.co.kr/) - 레벨별 코딩 테스트 문제
- [LeetCode](https://leetcode.com/) - 실전 코딩 인터뷰 문제

## 참고 자료

- "Introduction to Algorithms" by Thomas H. Cormen
- "Algorithms" by Robert Sedgewick
- "파이썬 알고리즘 인터뷰" by 박상길
- [VisuAlgo](https://visualgo.net/) - 알고리즘 시각화 사이트
