# ReduceWindows

[![Stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://jw3126.github.io/ReduceWindows.jl/stable/)
[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://jw3126.github.io/ReduceWindows.jl/dev/)
[![Build Status](https://github.com/jw3126/ReduceWindows.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/jw3126/ReduceWindows.jl/actions/workflows/CI.yml?query=branch%3Amain)
[![Coverage](https://codecov.io/gh/jw3126/ReduceWindows.jl/branch/main/graph/badge.svg)](https://codecov.io/gh/jw3126/ReduceWindows.jl)

Apply `reduce` over a sliding window.

# Usage
```julia
using ReduceWindows
x = [1,2,3,4,5]
reduce_window(+, x, (-1:1,))
# [3, 6, 9, 12, 9]
reduce_window(max, x, (-1:1,))
# [2, 3, 4, 5, 5]
reduce_window(min, x, (-3:0,))
# [1, 1, 1, 1, 2]

x = [1 2 3; 4 5 6]
reduce_window(*, x, (0:1,0:1))
# 40  180  18
# 20   30   6
```

# Speed
This package has very competitive performance, especially for large windows.
```julia
arr = randn(500,500)
window = (-50:50, -50:50)
using ImageFiltering: mapwindow
mapwindow(maximum, arr, window) # warmup
@showtime mapwindow(maximum, arr, window)

using ReduceWindows
reduce_window(max, arr, window) # warmup
@showtime reduce_window(max, arr, window)
```
```
mapwindow(maximum, arr, window): 2.075822 seconds (1.26 M allocations: 227.561 MiB, 0.76% gc time)
reduce_window(max, arr, window): 0.002320 seconds (14 allocations: 7.630 MiB)
```

# Alternatives

* [ImageFiltering.jl](https://github.com/JuliaImages/ImageFiltering.jl) much more features than
  this packge, but slow for large windows.
* [MeanFilters.jl](https://github.com/jw3126/MeanFilters.jl) fast, lightwight but very narrow usecase.
