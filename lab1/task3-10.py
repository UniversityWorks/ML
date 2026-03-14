def term(k, n, x):
    coef = n - k + 1
    numerator = (2 * k - 1)
    base = x + n - k + 1
    sign = (-1) ** (k + 1)

    if base == 0:
        return None
    denominator = base ** k
    return sign * coef * numerator / denominator

def sum_for(n, x):
    total = 0.0
    for k in range(1, n + 1):
        t = term(k, n, x)
        if t is None:
            print(f"  [for]  Помилка: знаменник (x+n-{k}+1) = {x + n - k + 1} = 0 при k={k}.")
            return False, None
        total += t
    return True, total

def sum_while(n, x):
    total = 0.0
    k = 1
    while k <= n:
        t = term(k, n, x)
        if t is None:
            print(f"  [while] Помилка: знаменник (x+n-{k}+1) = {x + n - k + 1} = 0 при k={k}.")
            return False, None
        total += t
        k += 1
    return True, total

def sum_recursive(n, x, k=1, accumulator=0.0):
    if k > n:
        return True, accumulator
    t = term(k, n, x)
    if t is None:
        print(f"  [рекурсія] Помилка: знаменник (x+n-{k}+1) = {x + n - k + 1} = 0 при k={k}.")
        return False, None
    return sum_recursive(n, x, k + 1, accumulator + t)

def main():
    print("=" * 60)
    print("  Обчислення суми n елементів виразу:")
    print("  S = n·1/(x+n) - (n-1)·3/pow((x+n-1),2) + (n-2)·5/pow((x+n-2),3) - ...")
    print("=" * 60)

    while True:
        try:
            n = int(input("Введіть натуральне число n (n Є N): "))
            if n < 1:
                print("  n повинно бути натуральним числом (≥ 1). Спробуйте ще раз.")
                continue
            break
        except ValueError:
            print("  Помилка: введіть ціле число.")

    while True:
        try:
            x = float(input("Введіть дійсне число x (x ∈ R): "))
            break
        except ValueError:
            print("  Помилка: введіть числове значення.")

    print()
    print(f"  n = {n},  x = {x}")
    print("-" * 60)

    ok_for, result_for = sum_for(n, x)
    if ok_for:
        print(f"  [for]       S = {result_for:.8f}  [x]")
    else:
        print("  [for]       Обчислення неможливе.")

    ok_while, result_while = sum_while(n, x)
    if ok_while:
        print(f"  [while]     S = {result_while:.8f}  [x]")
    else:
        print("  [while]     Обчислення неможливе.")

    import sys
    sys.setrecursionlimit(max(1000, n + 100))
    ok_rec, result_rec = sum_recursive(n, x)
    if ok_rec:
        print(f"  [рекурсія]  S = {result_rec:.8f}  [x]")
    else:
        print("  [рекурсія]  Обчислення неможливе.")

    print("-" * 60)
    if ok_for and ok_while and ok_rec:
        print("  Всі три методи дають однаковий результат — обчислення успішне.")

if __name__ == "__main__":
    main()
