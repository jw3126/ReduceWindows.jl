using ReduceWindows: reduce_window, reduce_window_naive
using ReduceWindows: fastmin, fastmax
using ReduceWindows
using Test
using Random: Xoshiro
using OffsetArrays

@testset "Digits" begin
    Digits = ReduceWindows.Digits
    for x in 1:100
        digits::Digits = @inferred Digits(x)
        @test Base.digits(x,base=2) == collect(digits)
        @test collect(digits) == [d for d in digits]
        n = length(collect(digits))
        @test collect(digits) == [Digits(x)[i] for i in 1:n]
    end
    @test length(Digits(1)) == 1
    @test length(Digits(2)) == 2
    @test length(Digits(3)) == 2
    @test length(Digits(4)) == 3
    for k in 1:60
        @test length(Digits(2^k+1)) == k+1
        @test length(Digits(2^k)) == k+1
        @test length(Digits(2^k-1)) == k
    end
end

@testset "1d explicit" begin
    @test reduce_window(+, 1:4, (-2:1,)) == [3, 6, 10, 9]
    @test reduce_window_naive(+, 1:4, (-2:1,)) == [3, 6, 10, 9]

    @test reduce_window(+, [1], (-1:0,)) == [1]
    @test reduce_window_naive(+, [1], (-1:0,)) == [1]

    @test reduce_window(+, [1], (-9:1,)) == [1]
    @test reduce_window_naive(+, [1], (-9:1,)) == [1]

    @test reduce_window(+, Float64[], (-2:1,)) == Float64[]
    @test reduce_window_naive(+, Float64[], (-2:1,)) == Float64[]


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

    @test reduce_window(+, OffsetArray([1,2,3,4],-2:1), (-2:1,)) == OffsetArray([3, 6, 10, 9], -2:1)
    @test reduce_window_naive(+, OffsetArray([1,2,3,4],-2:1), (-2:1,)) == OffsetArray([3, 6, 10, 9], -2:1)
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

@testset "eltype" begin
    @test typeof(@inferred reduce_window(+, 1:4, (-2:1,))) == Vector{Int}
    @test typeof(@inferred reduce_window(max, Float32[1,2], (-2:1,))) == Vector{Float32}
end

@testset "nd fuzz" begin
    rng = Xoshiro(1)
    myadd(x,y) = x + y
    for _ in 1:100
        nd = rand(rng, 1:4)
        win = ntuple(nd) do _
            lo = rand(rng, -5:0)
            hi = rand(rng, 0:5)
            lo:hi
        end
        siz = Tuple(rand(rng, 0:5, nd))
        arr = randn(rng, siz...)
        op = rand(rng, [+, *, min, max, fastmin, fastmax, myadd])
        @test reduce_window(+, arr, win) ≈ reduce_window_naive(+, arr, win)

        oaxes = map(siz) do n
            start = rand(rng, -10:10)
            stop = start+n-1
            start:stop
        end
        oarr = OffsetArray(arr, oaxes)
        @test reduce_window(+, oarr, win) ≈ reduce_window_naive(+, oarr, win)
    end
end

# import JET
# @testset "JET" begin
# arr = randn(Xoshiro(1), 10, 11)
# JET.@test_opt target_modules=(ReduceWindows,) reduce_window(+,arr, (-2:2,-4:3))
# end
