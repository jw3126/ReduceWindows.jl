using ReduceWindows: reduce_window, reduce_window_naive
using Test
using Random: Xoshiro

f = +
arr = 1:4
window = (-2:1,)
@test reduce_window(f, arr, window) ≈ reduce_window_naive(f, arr, window)

rng = Xoshiro(3)

f = +
arr = 1:5
window = (-0:0,)
@test reduce_window(f, arr, window) ≈ arr
@test reduce_window_naive(f, arr, window) ≈ arr
@test reduce_window(f, arr, window) ≈ reduce_window_naive(f, arr, window)

f = +
arr = randn(rng, 5)
window = (-1:1,)
@test reduce_window(f, arr, window) ≈ reduce_window_naive(f, arr, window)


f = +
arr = randn(rng, 5)
window = (-1:3,)
@test reduce_window(f, arr, window) ≈ reduce_window_naive(f, arr, window)

f = +
arr = randn(rng, 5)
window = (-10:0,)
@test reduce_window(f, arr, window) ≈ reduce_window_naive(f, arr, window)

f = +
arr = randn(rng, 5)
window = (0:4,)
@test reduce_window(f, arr, window) ≈ reduce_window_naive(f, arr, window)
