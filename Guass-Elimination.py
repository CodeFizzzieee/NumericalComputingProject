import numpy as np
import matplotlib.pyplot as plt
import time

def gauss_elimination(a, b):
    n = len(b)
    for i in range(n):
        for j in range(i+1, n):
            factor = a[j][i] / a[i][i]
            for k in range(i, n):
                a[j][k] -= factor * a[i][k]
            b[j] -= factor * b[i]
    x = np.zeros(n)
    for i in range(n-1, -1, -1):
        x[i] = b[i]
        for j in range(i+1, n):
            x[i] -= a[i][j] * x[j]
        x[i] /= a[i][i]
    return x

# Get user input for the system of equations
n = int(input("Enter the number of variables: "))

a = np.zeros((n, n))
b = np.zeros(n)

print("Enter the coefficients of the matrix row-wise:")
for i in range(n):
    for j in range(n):
        a[i][j] = float(input(f'a[{i}][{j}]: '))

print("Enter the constants of the matrix:")
for i in range(n):
    b[i] = float(input(f'b[{i}]: '))

start_time = time.perf_counter()
x = gauss_elimination(a, b)
end_time = time.perf_counter()
execution_time = end_time - start_time

print("Solution:")
for i in range(n):
    print(f'x[{i}] = {x[i]}')

print("\nExecution Time: {:.6f} seconds".format(execution_time))

# Plotting
plt.figure(figsize=(8, 6))
variables = range(n)
plt.scatter(variables, x, color='red', label="Data Points")
plt.plot(variables, x, color='green', alpha=0.7)
plt.xlabel("Variable")
plt.ylabel("Value")
plt.title("Gauss Elimination Method Solution")
plt.grid(True)
plt.show()
