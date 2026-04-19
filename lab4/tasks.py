import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib
matplotlib.use("Qt5Agg")
import matplotlib.pyplot as plt

def task1():
    M = int(input("Введіть M: "))
    N = int(input("Введіть N: "))
    matrix = np.full((M, N), 0.5)
    for i in range(min(M, N)):
        matrix[i][i] = -1
    print(matrix)

def task2():
    M = int(input("Введіть M: "))
    N = int(input("Введіть N: "))
    matrix = np.full((M, N), 0.5)
    for i in range(min(M, N)):
        matrix[i][i] = -1

    r1 = int(input("Рядок лівого верхнього кута: "))
    c1 = int(input("Стовпець лівого верхнього кута: "))
    r2 = int(input("Рядок правого нижнього кута: "))
    c2 = int(input("Стовпець правого нижнього кута: "))

    submatrix = matrix[r1:r2+1, c1:c2+1]
    print(submatrix)

def task3():
    a = np.array(list(map(int, input("Введіть перший масив через пробіл: ").split())))
    idx = np.array(list(map(int, input("Введіть індекси через пробіл: ").split())))
    result = a[idx]
    print(result)

def task4():
    a = np.array(list(map(float, input("Введіть масив через пробіл: ").split())))
    val = float(input("Введіть значення: "))
    choice = input("Більші чи менші? (b/m): ")
    if choice == "b":
        print(a[a > val])
    else:
        print(a[a < val])

def task5():
    K = int(input("Введіть K: "))
    N = 10
    matrix = np.zeros((K, K))
    for i in range(K):
        matrix[i][K - 1 - i] = N
    print(matrix)

def task6():
    N = 10
    matrix = np.zeros((7, 7), dtype=int)
    positions = [
        (0, 3),
        (1, 2), (1, 4),
        (2, 1), (2, 5),
        (3, 0), (3, 6),
        (4, 1), (4, 5),
        (5, 2), (5, 4),
        (6, 3)
    ]
    for r, c in positions:
        matrix[r][c] = N
    print(matrix)

def task7():
    K = int(input("Введіть K: "))
    N = 10
    matrix = np.zeros((K, K), dtype=int)
    matrix[1::2, ::2] = N  
    matrix[::2, 1::2] = N 
    print(matrix)

def task8():
    K = int(input("Введіть K (кількість елементів): "))
    L = float(input("Введіть L (мінімум діапазону): "))
    N = 10

    a = np.random.uniform(L, N, K)

    x_min = a.min()
    x_max = a.max()
    normalized = (a - x_min) / (x_max - x_min)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    ax1.bar(range(K), a, color="steelblue")
    ax1.set_title("Початковий масив")
    ax1.set_xlabel("Індекс")
    ax1.set_ylabel("Значення")

    ax2.bar(range(K), normalized, color="darkorange")
    ax2.set_title("Нормалізований масив")
    ax2.set_xlabel("Індекс")
    ax2.set_ylabel("Значення")

    plt.tight_layout()
    plt.savefig("task8_plot.png")
    print("Графік збережено у файл task8_plot.png")
    plt.close()

def task9():
    K = int(input("Введіть K (рядки): "))
    L = int(input("Введіть L (стовпці): "))

    matrix = np.random.randint(0, 256, (K, L))
    print("Початкова матриця:")
    print(matrix)

    smoothed = matrix.copy().astype(float)

    smoothed[1:-1, 1:-1] = (
        matrix[0:-2, 0:-2] + matrix[0:-2, 1:-1] + matrix[0:-2, 2:] +
        matrix[1:-1, 0:-2] + matrix[1:-1, 1:-1] + matrix[1:-1, 2:] +
        matrix[2:,   0:-2] + matrix[2:,   1:-1] + matrix[2:,   2:]
    ) / 9

    print("Згладжена матриця:")
    print(np.round(smoothed, 2))

def task10():
    x1 = np.linspace(-0.99, 3.99, 1000) 
    with np.errstate(divide='ignore', invalid='ignore'):
        y1 = (5 * np.sqrt(x1 + 1)) / np.tan(np.pi / 2 - x1)
    y1 = np.where(np.abs(y1) > 100, np.nan, y1)
 
    x2 = np.linspace(5.01, 99.99, 5000)
    with np.errstate(divide='ignore', invalid='ignore'):
        cos_x = np.cos(x2)
        ln_val = np.log(x2**2 - 36)
        y2 = np.where(cos_x >= 0, np.sqrt(cos_x) / ln_val, np.nan)
    y2 = np.where(np.abs(y2) > 100, np.nan, y2)
 
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
 
    ax1.plot(x1, y1, color="steelblue")
    ax1.axhline(0, color="black", linewidth=0.5)
    ax1.set_title("Гілка 1: x < 4")
    ax1.set_xlabel("x")
    ax1.set_ylabel("y")
    ax1.set_ylim(-20, 20)
    ax1.grid(True)
 
    ax2.plot(x2, y2, color="darkorange")
    ax2.axhline(0, color="black", linewidth=0.5)
    ax2.set_title("Гілка 2: 5 < x < 100")
    ax2.set_xlabel("x")
    ax2.set_ylabel("y")
    ax2.set_ylim(-1, 1)
    ax2.grid(True)
 
    fig.suptitle("Графік функції (варіант 10)")
    plt.tight_layout()
    plt.show()


tasks = {
    "1": ("Матриця MxN (діагональ -1, решта 0.5)", task1),
    "2": ("Підматриця за індексами кутів", task2),
    "3": ("Елементи масиву за індексами з другого масиву", task3),
    "4": ("Елементи більші/менші заданого значення", task4),
    "5": ("Матриця KxK з побічною діагоналлю", task5),
    "6": ("Масив 7x7 з X-патерном", task6),
    "7": ("Матриця KxK у шаховому порядку", task7),
    "8": ("K випадкових чисел, нормалізація, діаграма", task8),
    "9": ("Матриця KxL, згладжування без циклів", task9),
    "10": ("Графік функції", task10),
}


while True:
    print("Оберіть завдання:")
    for key, (name, _) in tasks.items():
        print(f"  {key}. {name}")
    print("  0. Вихід")
    print()

    choice = input("Ваш вибір: ").strip()

    if choice == "0":
        break
    elif choice in tasks:
        print()
        print(f" Завдання {choice}: {tasks[choice][0]}")
        tasks[choice][1]()
        print()
    else:
        print("Невірний вибір, спробуйте ще раз.")
        print()
