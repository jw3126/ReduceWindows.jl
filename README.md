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
