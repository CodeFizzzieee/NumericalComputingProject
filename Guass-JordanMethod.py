import numpy as np
import matplotlib.pyplot as plt
import time

def gauss_jordan(a, b):
    n = len(b)
    for i in range(n):
        # Partial pivoting to avoid division by zero
        if a[i, i] == 0:
            for k in range(i+1, n):
                if a[k, i] != 0:
                    a[[i, k]] = a[[k, i]]
                    b[[i, k]] = b[[k, i]]
                    break

        # Normalize the pivot row
        b[i] = b[i] / a[i, i]
        a[i] = a[i] / a[i, i]

        # Eliminate the column entries
        for j in range(n):
            if i != j:
                factor = a[j, i]
                a[j] -= factor * a[i]
                b[j] -= factor * b[i]
    return b

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
x = gauss_jordan(a, b)
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
plt.plot(variables, x, color='blue', alpha=0.7)
plt.xlabel("Variable")
plt.ylabel("Value")
plt.title("Gauss-Jordan Method Solution")
plt.grid(True)
plt.legend()
plt.show()
