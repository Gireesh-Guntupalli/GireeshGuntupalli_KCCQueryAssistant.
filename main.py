import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, Optional


def least_squares_regression(
    X: np.ndarray, y: np.ndarray
) -> Tuple[np.ndarray, float, float]:
    """
    Calculate least squares regression coefficients for linear regression.

    Args:
        X (np.ndarray): Feature matrix (n_samples, n_features)
        y (np.ndarray): Target values (n_samples,)

    Returns:
        Tuple containing:
        - coefficients (np.ndarray): Regression coefficients [intercept, slope(s)]
        - r_squared (float): R-squared value
        - mse (float): Mean squared error
    """
    # Add bias term (intercept) to X
    X_with_bias = np.column_stack([np.ones(X.shape[0]), X])

    # Calculate coefficients using normal equation: β = (X^T X)^(-1) X^T y
    XtX = X_with_bias.T @ X_with_bias
    Xty = X_with_bias.T @ y

    # Solve the system using numpy's linear algebra solver (more stable than inverse)
    coefficients = np.linalg.solve(XtX, Xty)

    # Make predictions
    y_pred = X_with_bias @ coefficients

    # Calculate R-squared
    ss_res = np.sum((y - y_pred) ** 2)  # Sum of squares of residuals
    ss_tot = np.sum((y - np.mean(y)) ** 2)  # Total sum of squares
    r_squared = 1 - (ss_res / ss_tot)

    # Calculate Mean Squared Error
    mse = np.mean((y - y_pred) ** 2)

    return coefficients, r_squared, mse


def generate_dummy_dataset(
    n_samples: int = 100, noise_level: float = 0.1, random_seed: Optional[int] = 42
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate a dummy dataset for testing least squares regression.

    Args:
        n_samples (int): Number of samples to generate
        noise_level (float): Amount of noise to add to the data
        random_seed (Optional[int]): Random seed for reproducibility

    Returns:
        Tuple of (X, y) where X is features and y is target
    """
    if random_seed is not None:
        np.random.seed(random_seed)

    # Generate feature data
    X = np.random.uniform(-5, 5, (n_samples, 1))

    # True relationship: y = 2.5 * x + 1.0 + noise
    true_slope = 2.5
    true_intercept = 1.0

    y = (
        true_intercept
        + true_slope * X.flatten()
        + np.random.normal(0, noise_level, n_samples)
    )

    return X, y


def plot_regression_results(
    X: np.ndarray, y: np.ndarray, coefficients: np.ndarray, r_squared: float, mse: float
) -> None:
    """
    Plot the regression results with the fitted line.

    Args:
        X (np.ndarray): Feature matrix
        y (np.ndarray): Target values
        coefficients (np.ndarray): Regression coefficients [intercept, slope]
        r_squared (float): R-squared value
        mse (float): Mean squared error
    """
    plt.figure(figsize=(10, 6))

    # Plot data points
    plt.scatter(X, y, alpha=0.6, color="blue", label="Data points")

    # Plot regression line
    X_line = np.linspace(X.min(), X.max(), 100).reshape(-1, 1)
    y_line = coefficients[0] + coefficients[1] * X_line.flatten()
    plt.plot(X_line, y_line, color="red", linewidth=2, label="Fitted line")

    # Add labels and title
    plt.xlabel("X (Feature)")
    plt.ylabel("y (Target)")
    plt.title(f"Least Squares Regression\nR² = {r_squared:.4f}, MSE = {mse:.4f}")
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Add equation text
    equation = f"y = {coefficients[0]:.2f} + {coefficients[1]:.2f}x"
    plt.text(
        0.05,
        0.95,
        equation,
        transform=plt.gca().transAxes,
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.8),
    )

    plt.tight_layout()
    plt.show()


def main():
    """
    Main function to demonstrate least squares regression with dummy data.
    """
    print("🔹 Least Squares Regression Demo")
    print("=" * 40)

    # Generate dummy dataset
    print("📊 Generating dummy dataset...")
    X, y = generate_dummy_dataset(n_samples=100, noise_level=0.5, random_seed=42)
    print(f"   Dataset shape: X={X.shape}, y={y.shape}")

    # Perform least squares regression
    print("\n🔍 Performing least squares regression...")
    coefficients, r_squared, mse = least_squares_regression(X, y)

    # Display results
    print("\n📈 Results:")
    print(f"   Intercept: {coefficients[0]:.4f}")
    print(f"   Slope: {coefficients[1]:.4f}")
    print(f"   R-squared: {r_squared:.4f}")
    print(f"   MSE: {mse:.4f}")

    # Create equation string
    equation = f"y = {coefficients[0]:.2f} + {coefficients[1]:.2f}x"
    print(f"   Equation: {equation}")

    # Plot results
    print("\n📊 Plotting results...")
    plot_regression_results(X, y, coefficients, r_squared, mse)

    # Example prediction
    print("\n🔮 Example predictions:")
    test_values = np.array([[0], [1], [2], [-1]])
    for test_x in test_values:
        pred_y = coefficients[0] + coefficients[1] * test_x[0]
        print(f"   x = {test_x[0]:2.0f} → y = {pred_y:.2f}")


if __name__ == "__main__":
    main()
