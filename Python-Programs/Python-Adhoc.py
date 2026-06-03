# Python Data Types Demo
# Day 1 Teaching Script

print("=== Numbers ===")
age = 25                # int
pi = 3.14159            # float
z = 2 + 3j              # complex
print("Integer:", age, type(age))
print("Float:", pi, type(pi))
print("Complex:", z, type(z))

print("\n=== Strings ===")
name = "Alice"
greeting = 'Hello'
print("String:", name, type(name))
print("Uppercase:", name.upper())
print("Repeat:", name * 2)

print("\n=== Booleans ===")
is_student = True
passed_exam = False
print("Boolean True:", is_student, type(is_student))
print("Boolean False:", passed_exam, type(passed_exam))
print("Comparison (5 > 3):", 5 > 3)

print("\n=== Lists ===")
fruits = ["apple", "banana", "cherry"]
fruits.append("mango")
print("List:", fruits, type(fruits))

print("\n=== Tuples ===")
coordinates = (10, 20)
print("Tuple:", coordinates, type(coordinates))

print("\n=== Range ===")
numbers = range(5)
print("Range:", list(numbers), type(numbers))

print("\n=== Sets ===")
colors = {"red", "blue", "green"}
colors.add("yellow")
print("Set:", colors, type(colors))

print("\n=== Frozen Set ===")
frozen_colors = frozenset(["red", "blue", "green"])
print("Frozen Set:", frozen_colors, type(frozen_colors))
