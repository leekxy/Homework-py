# Домашнее задание 1
# Тема: Векторы — нормы, косинусная близость,
# проекция, ортогонализация Грама–Шмидта и QR-разложение
#
# Важное условие: основные математические операции реализованы вручную.
# Не используются np.linalg.norm(), np.dot(), np.cross(), np.linalg.qr()
# и готовые функции для вычисления косинусной близости, проекции,
# Грама–Шмидта или QR-разложения.
#
# Для задания 2 используется готовая библиотека эмбеддингов,
# как непосредственно требуется в условии задания.

import random
import math


# ============================================================
# ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ
# ============================================================

def scalar_product(a, b):
    """Скалярное произведение двух векторов."""
    s = 0.0
    for i in range(len(a)):
        s += a[i] * b[i]
    return s


def vector_length(a):
    """Евклидова длина вектора."""
    return math.sqrt(scalar_product(a, a))


def vector_subtract(a, b):
    """a - b."""
    result = []
    for i in range(len(a)):
        result.append(a[i] - b[i])
    return result


def vector_add(a, b):
    """a + b."""
    result = []
    for i in range(len(a)):
        result.append(a[i] + b[i])
    return result


def multiply_vector_by_scalar(a, c):
    """Умножение вектора на число."""
    result = []
    for x in a:
        result.append(c * x)
    return result


def vector_norm_l1(a):
    """Норма L1 = сумма модулей координат."""
    s = 0.0
    for x in a:
        s += abs(x)
    return s


def vector_norm_l2(a):
    """Норма L2 = sqrt(sum(x_i^2))."""
    s = 0.0
    for x in a:
        s += x * x
    return math.sqrt(s)


def vector_norm_linf(a):
    """Норма L_inf = максимальный модуль координаты."""
    maximum = 0.0
    for x in a:
        if abs(x) > maximum:
            maximum = abs(x)
    return maximum


def cosine_similarity(a, b):
    """
    Косинусная близость:
    cos(a,b) = (a*b) / (|a|*|b|)
    """
    denominator = vector_length(a) * vector_length(b)

    if denominator == 0:
        raise ValueError("Косинусная близость для нулевого вектора не определена.")

    return scalar_product(a, b) / denominator


def vector_projection(x, y):
    """
    Векторная проекция x на y:
    proj_y(x) = ((x*y)/(y*y)) * y
    """
    denominator = scalar_product(y, y)

    if denominator == 0:
        raise ValueError("Нельзя проектировать на нулевой вектор.")

    coefficient = scalar_product(x, y) / denominator
    return multiply_vector_by_scalar(y, coefficient)


def determinant_3x3(matrix):
    """Определитель матрицы 3x3 вручную."""
    a = matrix[0][0]
    b = matrix[0][1]
    c = matrix[0][2]
    d = matrix[1][0]
    e = matrix[1][1]
    f = matrix[1][2]
    g = matrix[2][0]
    h = matrix[2][1]
    i = matrix[2][2]

    return (
        a * (e * i - f * h)
        - b * (d * i - f * g)
        + c * (d * h - e * g)
    )


def print_vector(name, v):
    print(name, "=", [round(x, 6) for x in v])


def print_matrix(name, matrix):
    print(name, "=")
    for row in matrix:
        print("   ", [round(x, 6) for x in row])


# ============================================================
# 1. НОРМЫ L1, L2, L_inf
# ============================================================

print("=" * 70)
print("ЗАДАНИЕ 1. НОРМЫ ВЕКТОРА")
print("=" * 70)

# Произвольный вектор размерности n
v1 = [3, -4, 2, -7, 5]

print_vector("Вектор v", v1)
print("L1  =", vector_norm_l1(v1))
print("L2  =", round(vector_norm_l2(v1), 6))
print("L_inf =", vector_norm_linf(v1))

# Для v = [3, -4, 2, -7, 5]:
# L1 = 21
# L2 = sqrt(103)
# L_inf = 7


# ============================================================
# 2. ЭМБЕДДИНГИ И КОСИНУСНАЯ БЛИЗОСТЬ
# ============================================================

print("\n" + "=" * 70)
print("ЗАДАНИЕ 2. ЭМБЕДДИНГИ")
print("=" * 70)

print("""
Для задания используется готовая библиотека Gensim и
предобученная модель glove-wiki-gigaword-50.

Сам поиск ближайшего и наиболее далекого слова выполняется
ВРУЧНУЮ: для каждого слова из словаря считается косинусная
близость нашей функцией cosine_similarity().

При первом запуске Gensim может скачать модель из Интернета.
""")

try:
    import gensim.downloader as api

    # Предобученная модель словесных эмбеддингов
    model = api.load("glove-wiki-gigaword-50")

    target_word = "computer"

    if target_word not in model.key_to_index:
        raise ValueError(
            "Слова '" + target_word + "' нет в словаре модели."
        )

    target_vector = model[target_word]

    nearest_word = None
    nearest_value = -float("inf")

    farthest_word = None
    farthest_value = float("inf")

    # Не используем model.most_similar():
    # косинусную близость и поиск экстремумов делаем вручную.
    for word in model.key_to_index:
        # Само с собой сравнивать не нужно.
        if word == target_word:
            continue

        current_vector = model[word]

        # Преобразуем в обычный список чисел
        current_list = []
        for x in current_vector:
            current_list.append(float(x))

        target_list = []
        for x in target_vector:
            target_list.append(float(x))

        similarity = cosine_similarity(target_list, current_list)

        if similarity > nearest_value:
            nearest_value = similarity
            nearest_word = word

        if similarity < farthest_value:
            farthest_value = similarity
            farthest_word = word

    print("Исходное слово:", target_word)
    print("Наиболее близкое слово:", nearest_word)
    print("Косинусная близость:", round(nearest_value, 6))
    print("Наиболее далекое слово:", farthest_word)
    print("Косинусная близость:", round(farthest_value, 6))

except ImportError:
    print("Для задания 2 установите Gensim:")
    print("pip install gensim")
except Exception as error:
    print("Не удалось выполнить задание 2.")
    print("Причина:", error)
    print(
        "Если модель не загрузилась, установите Gensim и "
        "обеспечьте доступ к Интернету при первом запуске."
    )


# ============================================================
# 3. ПРОЕКЦИЯ ВЕКТОРА x НА ВЕКТОР y
# ============================================================

print("\n" + "=" * 70)
print("ЗАДАНИЕ 3. ВЕКТОРНАЯ ПРОЕКЦИЯ")
print("=" * 70)

x = [4, 2, 3]
y = [1, 3, 2]

projection = vector_projection(x, y)

print_vector("x", x)
print_vector("y", y)
print_vector("proj_y(x)", projection)

print(
    "Проверка: скалярное произведение "
    "(x - proj_y(x)) и y =",
    round(
        scalar_product(
            vector_subtract(x, projection),
            y
        ),
        10
    )
)

# Построение графика выполняется только для визуализации.
# Математические расчеты выше сделаны вручную.
try:
    import matplotlib.pyplot as plt

    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")

    # Вектор x
    ax.quiver(
        0, 0, 0,
        x[0], x[1], x[2],
        arrow_length_ratio=0.08,
        label="x"
    )

    # Вектор y
    ax.quiver(
        0, 0, 0,
        y[0], y[1], y[2],
        arrow_length_ratio=0.08,
        label="y"
    )

    # Проекция x на y
    ax.quiver(
        0, 0, 0,
        projection[0], projection[1], projection[2],
        arrow_length_ratio=0.08,
        label="proj_y(x)"
    )

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_title("Векторы x, y и проекция x на y")
    ax.legend()

    plt.show()

except ImportError:
    print("Для построения графика установите matplotlib:")
    print("pip install matplotlib")


# ============================================================
# 4. ОРТОГОНАЛИЗАЦИЯ ГРАМА–ШМИДТА
# ============================================================

print("\n" + "=" * 70)
print("ЗАДАНИЕ 4. ГРАМ–ШМИДТ")
print("=" * 70)

# Фиксируем генератор, чтобы результаты можно было воспроизвести.
random.seed(42)


def generate_basis():
    """
    Генерирует три случайных линейно независимых
    и попарно неортогональных вектора в R^3.
    """
    while True:
        v1 = [
            random.randint(-5, 5),
            random.randint(-5, 5),
            random.randint(-5, 5)
        ]

        v2 = [
            random.randint(-5, 5),
            random.randint(-5, 5),
            random.randint(-5, 5)
        ]

        v3 = [
            random.randint(-5, 5),
            random.randint(-5, 5),
            random.randint(-5, 5)
        ]

        # Нулевые векторы исключаем.
        if vector_length(v1) == 0:
            continue
        if vector_length(v2) == 0:
            continue
        if vector_length(v3) == 0:
            continue

        # Линейная независимость:
        # векторы образуют матрицу, определитель которой не равен 0.
        matrix = [
            [v1[0], v2[0], v3[0]],
            [v1[1], v2[1], v3[1]],
            [v1[2], v2[2], v3[2]]
        ]

        if determinant_3x3(matrix) == 0:
            continue

        # Проверяем, что векторы не ортогональны попарно.
        if scalar_product(v1, v2) == 0:
            continue
        if scalar_product(v1, v3) == 0:
            continue
        if scalar_product(v2, v3) == 0:
            continue

        return [v1, v2, v3]


def gram_schmidt(vectors):
    """
    Ортогонализация Грама–Шмидта.

    Для первого:
        u1 = v1

    Для второго:
        u2 = v2 - proj_u1(v2)

    Для третьего:
        u3 = v3 - proj_u1(v3) - proj_u2(v3)

    Затем каждый u нормируется:
        q_i = u_i / ||u_i||
    """
    u = []

    for v in vectors:
        current = []

        # Копируем текущий вектор
        for value in v:
            current.append(float(value))

        # Вычитаем проекции на все уже полученные
        # ортогональные векторы.
        for previous in u:
            denominator = scalar_product(previous, previous)

            coefficient = (
                scalar_product(v, previous)
                / denominator
            )

            projection = multiply_vector_by_scalar(
                previous,
                coefficient
            )

            current = vector_subtract(
                current,
                projection
            )

        u.append(current)

    # Нормирование ортогональных векторов.
    q = []

    for vector in u:
        length = vector_length(vector)

        if length == 0:
            raise ValueError(
                "Получился нулевой вектор. "
                "Базис должен быть линейно независимым."
            )

        normalized = []
        for value in vector:
            normalized.append(value / length)

        q.append(normalized)

    return q


v_basis = generate_basis()

print("Случайно сгенерированный базис v:")
for i in range(3):
    print_vector("v" + str(i + 1), v_basis[i])

q_basis = gram_schmidt(v_basis)

print("\nОртонормированный базис q:")
for i in range(3):
    print_vector("q" + str(i + 1), q_basis[i])

print("\nПроверка единичной длины:")
for i in range(3):
    print(
        "||q" + str(i + 1) + "|| =",
        round(vector_length(q_basis[i]), 6)
    )

print("\nПроверка ортогональности:")
print("q1 * q2 =", round(scalar_product(q_basis[0], q_basis[1]), 10))
print("q1 * q3 =", round(scalar_product(q_basis[0], q_basis[2]), 10))
print("q2 * q3 =", round(scalar_product(q_basis[1], q_basis[2]), 10))


# ============================================================
# 5. QR-РАЗЛОЖЕНИЕ A = QR
# ============================================================

print("\n" + "=" * 70)
print("ЗАДАНИЕ 5. QR-РАЗЛОЖЕНИЕ")
print("=" * 70)

# A — матрица, столбцами которой являются исходные v.
A = [
    [v_basis[0][0], v_basis[1][0], v_basis[2][0]],
    [v_basis[0][1], v_basis[1][1], v_basis[2][1]],
    [v_basis[0][2], v_basis[1][2], v_basis[2][2]]
]

# Q — матрица, столбцами которой являются q.
Q = [
    [q_basis[0][0], q_basis[1][0], q_basis[2][0]],
    [q_basis[0][1], q_basis[1][1], q_basis[2][1]],
    [q_basis[0][2], q_basis[1][2], q_basis[2][2]]
]


def transpose(matrix):
    """Транспонирование матрицы вручную."""
    rows = len(matrix)
    cols = len(matrix[0])

    result = []

    for j in range(cols):
        row = []
        for i in range(rows):
            row.append(matrix[i][j])
        result.append(row)

    return result


def matrix_multiply(A, B):
    """Умножение матриц вручную."""
    rows_a = len(A)
    cols_a = len(A[0])
    cols_b = len(B[0])

    result = []

    for i in range(rows_a):
        row = []

        for j in range(cols_b):
            value = 0.0

            for k in range(cols_a):
                value += A[i][k] * B[k][j]

            row.append(value)

        result.append(row)

    return result


def build_R(Q, A):
    """
    R = Q^T A.

    Так как Q ортонормирована, элементы:
        R_ij = q_i * v_j

    Для i > j значения должны быть близки к нулю.
    """
    QT = transpose(Q)
    return matrix_multiply(QT, A)


R = build_R(Q, A)

print_matrix("A", A)
print_matrix("Q", Q)
print_matrix("R", R)

# Проверяем A = QR
QR = matrix_multiply(Q, R)

print_matrix("Q * R", QR)

print("\nПроверка ошибки A - QR:")
max_error = 0.0

for i in range(3):
    for j in range(3):
        error = abs(A[i][j] - QR[i][j])

        if error > max_error:
            max_error = error

print("Максимальная абсолютная ошибка =", max_error)

# Проверяем, что R верхнетреугольная.
print("\nПроверка верхней треугольности R:")
print("R21 =", round(R[1][0], 10))
print("R31 =", round(R[2][0], 10))
print("R32 =", round(R[2][1], 10))


# ============================================================
# ИТОГ
# ============================================================

print("\n" + "=" * 70)
print("РАБОТА ЗАВЕРШЕНА")
print("=" * 70)

print("""
В работе:
1. Вручную вычислены нормы L1, L2 и L_inf.
2. Для эмбеддингов использована готовая библиотека Gensim,
   а косинусная близость и поиск экстремумов реализованы вручную.
3. Вручную вычислена векторная проекция x на y и построен график.
4. Реализована ортогонализация Грама–Шмидта без np.linalg.qr()
   и других готовых алгоритмов линейной алгебры.
5. На основе полученных базисов вручную построено QR-разложение
   A = QR и выполнена проверка результата.
""")
