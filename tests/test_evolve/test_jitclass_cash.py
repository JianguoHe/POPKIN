import numpy as np
import time
from numba import njit, float64
from numba.experimental import jitclass

# ============================================================
# 定义 jitclass
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
        """计算：value + sum(array) * multiplier"""
        s = 0.0
        for i in range(self.array.shape[0]):
            s += self.array[i]
        return self.value + s * multiplier



# ============================================================
# 场景1：内部创建 jitclass 实例的函数（你的场景）
# ============================================================
@njit(cache=True)
def function_with_jitclass_creation(n: int, multiplier: float):
    """每次调用内部创建 jitclass 实例"""
    arr = np.arange(n, dtype=np.float64)  # 0, 1, 2, ..., n-1
    obj = TestClass(100.0, arr)  # 创建 jitclass 实例
    # return obj.compute(multiplier)



# ============================================================
# 场景2：不涉及 jitclass 的普通函数（对照组）
# ============================================================
@njit(cache=True)
def function_without_jitclass(n: int, multiplier: float) -> float:
    """纯函数，不含 jitclass"""
    arr = np.arange(n, dtype=np.float64)
    s = 0.0
    for i in range(arr.shape[0]):
        s += arr[i]
    return 100.0 + s * multiplier






# ============================================================
# 测试函数
# ============================================================
def test_cache_behavior():
    print("=" * 70)
    print("测试：jitclass 与 cache=True 的兼容性")
    print("=" * 70)

    n = 10000
    multiplier = 2.5

    # --------------------------------------------------------
    # 测试1：包含 jitclass 创建的函数
    # --------------------------------------------------------
    print("\n[测试1] @njit(cache=True) 函数内部创建 jitclass 实例")
    print("-" * 50)

    # 第一次运行（应包含编译时间）
    start = time.perf_counter()
    function_with_jitclass_creation(n, multiplier)
    time1 = time.perf_counter() - start
    print(f"第1次调用: {time1 * 1000:.3f} ms, 结果:")

    # 第二次运行（同一进程，应极快）
    start = time.perf_counter()
    result2 = function_with_jitclass_creation(n, multiplier)
    time2 = time.perf_counter() - start
    print(f"第2次调用: {time2 * 1000:.3f} ms, 结果: {result2}")

    print(f"第2次/第1次耗时比: {time2 / time1:.3f}")

    # --------------------------------------------------------
    # 测试2：不涉及 jitclass 的函数（对照组）
    # --------------------------------------------------------
    print("\n[测试2] @njit(cache=True) 普通函数（对照组）")
    print("-" * 50)

    start = time.perf_counter()
    result3 = function_without_jitclass(n, multiplier)
    time3 = time.perf_counter() - start
    print(f"第1次调用: {time3 * 1000:.3f} ms, 结果: {result3}")

    start = time.perf_counter()
    result4 = function_without_jitclass(n, multiplier)
    time4 = time.perf_counter() - start
    print(f"第2次调用: {time4 * 1000:.3f} ms, 结果: {result4}")

    print(f"第2次/第1次耗时比: {time4 / time3:.3f}")

    # --------------------------------------------------------
    # 说明
    # --------------------------------------------------------
    print("\n" + "=" * 70)
    print("预期结果说明：")
    print("1. 场景1（含 jitclass）: 第2次调用不会显著快于第1次")
    print("   因为 jitclass 无法被正确缓存，每次都在重新编译。")
    print("2. 场景2（普通函数）: 第2次调用应明显快于第1次")
    print("   因为缓存机制正常工作。")
    print("=" * 70)


if __name__ == "__main__":
    # 注意：需要运行两次脚本才能观察跨进程的缓存行为
    # 第一次运行：两个场景都会有编译开销
    # 第二次运行：场景1仍然有编译开销，场景2直接加载缓存
    test_cache_behavior()