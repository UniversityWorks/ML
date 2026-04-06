
import random
import math

def task1():
    print("\n=== Завдання 1 ===")

    def fill_random_list(n, a, b):
        return [random.randint(a, b) for _ in range(n)]

    n = int(input("Введіть кількість елементів n: "))
    a = int(input("Введіть нижню межу a: "))
    b = int(input("Введіть верхню межу b: "))
    result = fill_random_list(n, a, b)
    print(f"Список: {result}")



def task2():
    print("\n=== Завдання 2 ===")

    def get_negatives_no_gen(lst):
        result = []
        for x in lst:
            if x < 0:
                result.append(x)
        return result

    def get_negatives_gen(lst):
        return [x for x in lst if x < 0]

    n = int(input("Введіть кількість елементів: "))
    a = [int(input(f"  a[{i}] = ")) for i in range(n)]

    print(f"Список a: {a}")
    print(f"Від'ємні (без генератора): {get_negatives_no_gen(a)}")
    print(f"Від'ємні (з генератором):  {get_negatives_gen(a)}")



def task3():
    print("\n=== Завдання 3 ===")

    def remove_element_no_gen(lst, n):
        result = []
        for x in lst:
            if x != n:
                result.append(x)
        return result

    def remove_element_gen(lst, n):
        return [x for x in lst if x != n]

    size = int(input("Введіть розмір списку: "))
    lst = [int(input(f"  lst[{i}] = ")) for i in range(size)]
    n = int(input("Введіть число n для видалення: "))

    print(f"Список: {lst}")
    print(f"Без {n} (без генератора): {remove_element_no_gen(lst, n)}")
    print(f"Без {n} (з генератором):  {remove_element_gen(lst, n)}")


def task4():
    print("\n=== Завдання 4 ===")

    def insert_before_index(lst, index, value):
        return lst[:index] + [value] + lst[index:]

    size = int(input("Введіть розмір списку: "))
    lst = [int(input(f"  lst[{i}] = ")) for i in range(size)]
    index = int(input("Введіть індекс: "))
    value = int(input("Введіть значення для вставки: "))

    result = insert_before_index(lst, index, value)
    print(f"Список після вставки: {result}")



def task5():
    print("\n=== Завдання 5 ===")

    groups = {
        "КН-11": {"count": 25, "head": "Іваненко Олег Петрович"},
        "КН-12": {"count": 22, "head": "Марченко Аліна Сергіївна"},
        "КН-21": {"count": 28, "head": "Бондаренко Микола Юрійович"},
    }

    def show_menu():
        print("\n--- Меню ---")
        print("1. Кількість студентів у групі")
        print("2. ПІБ старости групи")
        print("3. Список груп, де кількість студентів не перевищує значення")
        print("4. Кортеж груп, де кількість студентів не менше значення")
        print("5. Змінити кількість студентів у групі")
        print("6. Змінити ПІБ старости у групі")
        print("7. Додати нову групу")
        print("8. Видалити групу")
        print("9. Отримати множину ПІБ старост зазначених груп")
        print("10. Вийти з програми")

    while True:
        show_menu()
        choice = input("Ваш вибір: ").strip()

        if choice == "1":
            g = input("Введіть назву групи: ")
            if g in groups:
                print(f"Кількість студентів у {g}: {groups[g]['count']}")
            else:
                print("Групу не знайдено.")

        elif choice == "2":
            g = input("Введіть назву групи: ")
            if g in groups:
                print(f"Староста {g}: {groups[g]['head']}")
            else:
                print("Групу не знайдено.")

        elif choice == "3":
            limit = int(input("Введіть максимальну кількість студентів: "))
            result = [g for g, d in groups.items() if d["count"] <= limit]
            print(f"Групи з кількістю ≤ {limit}: {result}")

        elif choice == "4":
            limit = int(input("Введіть мінімальну кількість студентів: "))
            result = tuple(g for g, d in groups.items() if d["count"] >= limit)
            print(f"Групи з кількістю ≥ {limit}: {result}")

        elif choice == "5":
            g = input("Введіть назву групи: ")
            if g in groups:
                new_count = int(input("Нова кількість студентів: "))
                groups[g]["count"] = new_count
                print("Оновлено.")
            else:
                print("Групу не знайдено.")

        elif choice == "6":
            g = input("Введіть назву групи: ")
            if g in groups:
                new_head = input("Новий ПІБ старости: ")
                groups[g]["head"] = new_head
                print("Оновлено.")
            else:
                print("Групу не знайдено.")

        elif choice == "7":
            g = input("Назва нової групи: ")
            if g in groups:
                print("Така група вже існує.")
            else:
                count = int(input("Кількість студентів: "))
                head = input("ПІБ старости: ")
                groups[g] = {"count": count, "head": head}
                print("Групу додано.")

        elif choice == "8":
            g = input("Введіть назву групи для видалення: ")
            if g in groups:
                del groups[g]
                print("Групу видалено.")
            else:
                print("Групу не знайдено.")

        elif choice == "9":
            names_input = input("Введіть назви груп через пробіл: ").split()
            heads_set = set()
            for g in names_input:
                if g in groups:
                    heads_set.add(groups[g]["head"])
                else:
                    print(f"Групу {g} не знайдено.")
            print(f"Множина старост: {heads_set}")

        elif choice == "10":
            print("Вихід.")
            break
        else:
            print("Невірний вибір.")



def task6():
    print("\n=== Завдання 6 ===")
    print("Введіть n (максимальне задумане число):")
    n = int(input())

    questions = []
    line = input() 
    while line != "Все":
        parts = list(map(int, line.split()))
        answer = input().strip()
        questions.append((parts, answer))
        line = input()

    possible = set(range(1, n + 1))

    for nums, ans in questions:
        asked_set = set(nums)
        if ans == "Так":
            possible &= asked_set
        else:
            possible -= asked_set

    print(" ".join(map(str, sorted(possible))))


def task7():
    print("\n=== Завдання 7 (Варіант 11) ===")

    def prime_factors(n):
        factors = []
        d = 2
        while d * d <= n:
            while n % d == 0:
                factors.append(d)
                n //= d
            d += 1
        if n > 1:
            factors.append(n)
        return factors

    n = int(input("Введіть додатне ціле число: "))
    result = prime_factors(n)
    print(f"Прості множники {n}: {result}")




def main():
    tasks = {
        "1": task1,
        "2": task2,
        "3": task3,
        "4": task4,
        "5": task5,
        "6": task6,
        "7": task7,
    }

    print("Виберіть завдання (1–7) або 0 для виходу:")
    while True:
        choice = input("\nЗавдання: ").strip()
        if choice == "0":
            break
        elif choice in tasks:
            tasks[choice]()
        else:
            print("Невірний вибір. Введіть число від 1 до 7.")


if __name__ == "__main__":
    main()
