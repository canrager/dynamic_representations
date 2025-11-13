import torch
import matplotlib.pyplot as plt
from einops import rearrange

# Define the function
def f(x):
    return -2.47 * x ** -0.1 + 2.09

# Define the inverse function to solve for x given y
def inverse_f(y):
    # y = -2.47 * x^-0.1 + 2.09
    # y - 2.09 = -2.47 * x^-0.1
    # (y - 2.09) / -2.47 = x^-0.1
    # x = ((y - 2.09) / -2.47) ^ -10
    return ((y - 2.09) / -2.47) ** -10

# Set up the range for x
x_min, x_max = 1, 500
y_min, y_max = f(x_max), f(x_min)

# Create 5 equally-spaced y values
n_points = 5
y_values = torch.linspace(y_min, y_max, n_points)
x_values = torch.tensor([inverse_f(y.item()) for y in y_values])

# Create continuous curve for plotting
x_continuous = torch.linspace(x_min, x_max, 1000)
y_continuous = f(x_continuous)

# Print the x, y pairs
print("X values for equal steps in Y:")
for i, (x, y) in enumerate(zip(x_values, y_values)):
    print(f"Point {i+1}: x = {x:.2f}, y = {y:.4f}")

# Plot
plt.figure(figsize=(10, 6))
plt.plot(x_continuous, y_continuous, 'b-', linewidth=2, label='y = -2.47x^(-0.1) + 2.09')
plt.scatter(x_values, y_values, color='red', s=100, zorder=5, label='Equal Y-step points')

# Add horizontal lines to show equal spacing
for y in y_values:
    plt.axhline(y=y, color='gray', linestyle='--', alpha=0.3)

plt.xlabel('x')
plt.ylabel('y')
plt.title('Function with 5 Equally-Spaced Y Values')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('equal_steps.png', dpi=150)
print("\nPlot saved as 'equal_steps.png'")
plt.show()
