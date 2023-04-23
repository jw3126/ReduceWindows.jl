using ReduceWindows: reduce_window, reduce_window_naive
using ReduceWindows: fastmin, fastmax
using Test
using Random: Xoshiro

@testset "1d explicit" begin
    @test reduce_window(+, [1], (-1:0,)) == [1]
    @test reduce_window_naive(+, [1], (-1:0,)) == [1]

    @test reduce_window(+, [1], (-9:1,)) == [1]
    @test reduce_window_naive(+, [1], (-9:1,)) == [1]

    @test reduce_window(+, Float64[], (-2:1,)) == Float64[]
    @test reduce_window_naive(+, Float64[], (-2:1,)) == Float64[]

    @test reduce_window(+, 1:4, (-2:1,)) == [3, 6, 10, 9]
    @test reduce_window_naive(+, 1:4, (-2:1,)) == [3, 6, 10, 9]

    @test reduce_window(+, [1,3,4], (0:0,)) ≈ [1,3,4]
    @test reduce_window_naive(+, [1,3,4], (0:0,)) ≈ [1,3,4]

    @test reduce_window(+, [1,3,4], (-1:1,)) ≈ [4, 8, 7]
    @test reduce_window_naive(+, [1,3,4], (-1:1,)) ≈ [4, 8, 7]

    @test reduce_window(+, [1,2], (0:4,)) ≈ [3,2]
    @test reduce_window_naive(+, [1,2], (0:4,)) ≈ [3,2]

    @test reduce_window(max, [1,3,4], (-1:3,)) == [4,4,4]
    @test reduce_window_naive(max, [1,3,4], (-1:3,)) == [4,4,4]

    @test reduce_window(min, [5,3,4,1], (-10:0,)) == [5, 3, 3, 1]
    @test reduce_window_naive(min, [5,3,4,1], (-10:0,)) == [5, 3, 3, 1]

    @test reduce_window(min, [5,3,4,1], (-10:10,)) == [1,1,1,1]
    @test reduce_window_naive(min, [5,3,4,1], (-10:10,)) == [1,1,1,1]
end

@testset "2d explicit" begin
    @test reduce_window(+, [1 2; 3 4], (0:0,0:0)) == [1 2; 3 4]
    @test reduce_window_naive(+, [1 2; 3 4], (0:0,0:0)) == [1 2; 3 4]

    @test reduce_window(+, [1 2; 3 4], (0:1,0:0)) == [4 6; 3 4]
    @test reduce_window_naive(+, [1 2; 3 4], (0:1,0:0)) == [4 6; 3 4]

    @test reduce_window(+, [1 2; 3 4], (0:0,0:1)) == [3 2; 7 4]
    @test reduce_window_naive(+, [1 2; 3 4], (0:0,0:1)) == [3 2; 7 4]

    @test reduce_window(+, [1 2; 3 4], (0:1,0:1)) == [10 6; 7 4]
    @test reduce_window_naive(+, [1 2; 3 4], (0:1,0:1)) == [10 6; 7 4]

    @test reduce_window(+, [1 2; 3 4], (-10:11,-12:13)) == [10 10; 10 10]
    @test reduce_window_naive(+, [1 2; 3 4], (-10:11,-12:13)) == [10 10; 10 10]
end

@testset "1d fuzz" begin
    rng = Xoshiro(1)
    for _ in 1:100
        lo = rand(rng, -10:0)
        hi = rand(rng, 0:10)
        win = (lo:hi,)
        len = rand(rng, 0:20)
        arr = randn(rng, len)
        @test reduce_window(+, arr, win) ≈ reduce_window_naive(+, arr, win)
    end
end

@testset "2d fuzz" begin
    rng = Xoshiro(1)
    for _ in 1:100
        win = ntuple(2) do _
            lo = rand(rng, -10:0)
            hi = rand(rng, 0:10)
            lo:hi
        end
        siz = rand(rng, 0:20, 2)
        arr = randn(rng, siz...)
        @test reduce_window(+, arr, win) ≈ reduce_window_naive(+, arr, win)
    end
end

@testset "nd fuzz" begin
    rng = Xoshiro(1)
    for _ in 1:100
        nd = rand(rng, 1:4)
        win = ntuple(nd) do _
            lo = rand(rng, -5:0)
            hi = rand(rng, 0:5)
            lo:hi
        end
        siz = rand(rng, 0:5, nd)
        arr = randn(rng, siz...)
        op = rand(rng, [+, min, max, fastmin, fastmax])
        @test reduce_window(+, arr, win) ≈ reduce_window_naive(+, arr, win)
    end
end

# import JET
# @testset "JET" begin
# arr = randn(Xoshiro(1), 10, 11)
# JET.@test_opt target_modules=(ReduceWindows,) reduce_window(+,arr, (-2:2,-4:3))
# end
