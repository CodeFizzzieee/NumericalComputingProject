import numpy as np
import matplotlib.pyplot as plt
import time

def jacobi(a, b, tolerance=1e-10, max_iterations=1000):
    n = len(b)
    x = np.zeros(n)
    for k in range(max_iterations):
        x_new = np.copy(x)
        for i in range(n):
            sum_ax = np.dot(a[i, :], x) - a[i, i] * x[i]
            x_new[i] = (b[i] - sum_ax) / a[i, i]
        if np.linalg.norm(x_new - x, ord=np.inf) < tolerance:
            return x_new
        x = x_new
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
x = jacobi(a, b)
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
plt.plot(variables, x, color='red', alpha=0.7)
plt.xlabel("Variable")
plt.ylabel("Value")
plt.title("Jacobi Method Solution")
plt.grid(True)
plt.show()
