import numpy as np
import time
from numba import njit, float64
from numba.experimental import jitclass

# ============================================================
# Define the jitclass.
# ============================================================
spec = [
    ('value', float64),
    ('array', float64[:])
]



@jitclass(spec)
class TestClass:
    def __init__(self, value: float, arr: np.ndarray):
        self.value = value
        self.array = arr

    def compute(self, multiplier: float) -> float:
        """Compute value + sum(array) * multiplier."""
        s = 0.0
        for i in range(self.array.shape[0]):
            s += self.array[i]
        return self.value + s * multiplier



# ============================================================
# Scenario 1: a function that creates a jitclass instance internally.
# ============================================================
@njit(cache=True)
def function_with_jitclass_creation(n: int, multiplier: float):
    """Create a jitclass instance internally on each call."""
    arr = np.arange(n, dtype=np.float64)  # 0, 1, 2, ..., n-1
    obj = TestClass(100.0, arr)  # Create a jitclass instance.
    # return obj.compute(multiplier)



# ============================================================
# Scenario 2: a normal function without jitclass usage.
# ============================================================
@njit(cache=True)
def function_without_jitclass(n: int, multiplier: float) -> float:
    """Pure function without a jitclass."""
    arr = np.arange(n, dtype=np.float64)
    s = 0.0
    for i in range(arr.shape[0]):
        s += arr[i]
    return 100.0 + s * multiplier






# ============================================================
# Test function.
# ============================================================
def test_cache_behavior():
    print("=" * 70)
    print("Test: compatibility between jitclass and cache=True")
    print("=" * 70)

    n = 10000
    multiplier = 2.5

    # --------------------------------------------------------
    # Test 1: a function that creates a jitclass instance.
    # --------------------------------------------------------
    print("\n[Test 1] @njit(cache=True) function creates a jitclass instance internally")
    print("-" * 50)

    # First run, expected to include compilation time.
    start = time.perf_counter()
    function_with_jitclass_creation(n, multiplier)
    time1 = time.perf_counter() - start
    print(f"First call: {time1 * 1000:.3f} ms, result:")

    # Second run in the same process, expected to be very fast.
    start = time.perf_counter()
    result2 = function_with_jitclass_creation(n, multiplier)
    time2 = time.perf_counter() - start
    print(f"Second call: {time2 * 1000:.3f} ms, result: {result2}")

    print(f"Second/first call time ratio: {time2 / time1:.3f}")

    # --------------------------------------------------------
    # Test 2: a function without jitclass usage.
    # --------------------------------------------------------
    print("\n[Test 2] @njit(cache=True) normal function")
    print("-" * 50)

    start = time.perf_counter()
    result3 = function_without_jitclass(n, multiplier)
    time3 = time.perf_counter() - start
    print(f"First call: {time3 * 1000:.3f} ms, result: {result3}")

    start = time.perf_counter()
    result4 = function_without_jitclass(n, multiplier)
    time4 = time.perf_counter() - start
    print(f"Second call: {time4 * 1000:.3f} ms, result: {result4}")

    print(f"Second/first call time ratio: {time4 / time3:.3f}")

    # --------------------------------------------------------
    # Explanation.
    # --------------------------------------------------------
    print("\n" + "=" * 70)
    print("Expected behavior:")
    print("1. Scenario 1 (with jitclass): the second call is not much faster than the first")
    print("   because the jitclass cannot be cached properly and is recompiled each time.")
    print("2. Scenario 2 (normal function): the second call should be much faster than the first")
    print("   because the cache mechanism works normally.")
    print("=" * 70)


if __name__ == "__main__":
    # Note: run this script twice to observe cross-process cache behavior.
    # First run: both scenarios include compilation overhead.
    # Second run: scenario 1 still has compilation overhead, while scenario 2 loads from cache.
    test_cache_behavior()
