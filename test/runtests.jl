using ReduceWindows: reduce_window, reduce_window_naive
using ReduceWindows: along_axis!, calc_fwd!, calc_bwd!
using ReduceWindows: fastmin, fastmax
using ReduceWindows
using Test
using Random: Xoshiro
using OffsetArrays

function calc_fwd_free_monoid(;length, stride)
    inp = [[i] for i in 1:length]
    f = vcat
    out = fill(Int[], length)
    Dim = Val(1)
    calc_fwd!(f, out, inp, Dim, stride)
    return out
end

function calc_bwd_free_monoid(;length, stride)
    inp = [[i] for i in 1:length]
    f = vcat
    out = fill(Int[], length)
    Dim = Val(1)
    calc_bwd!(f, out, inp, Dim, stride)
    return out
end

function along_axis_free_monoid(; length, winaxis)
    inp = [[i] for i in 1:length]
    f = vcat
    out = fill(Int[], length)
    Dim = Val(1)
    deadpool = DeadPool(inp)
    along_axis!(f, out, inp, Dim, winaxis, deadpool)
    return out
end

@testset "along_axis_free_monoid" begin
    @test along_axis_free_monoid(; length = 5, winaxis = 0:1) == 
        [[1, 2], [2, 3], [3, 4], [4, 5], [5]]
    
    @test along_axis_free_monoid(; length = 5, winaxis = 0:2) == 
        [[1, 2, 3], [2, 3, 4], [3, 4, 5], [4, 5], [5]]
    
    @test along_axis_free_monoid(; length = 5, winaxis = -1:1) ==
        [[1, 2], [1, 2, 3], [2, 3, 4], [3, 4, 5], [4,5]]
    
    @test along_axis_free_monoid(; length = 5, winaxis = 0:1) == 
        [[1, 2], [2, 3], [3, 4], [4, 5], [5]]
    
    @test along_axis_free_monoid(; length = 5, winaxis = -1:0) == 
        [[1], [1, 2], [2, 3], [3, 4], [4, 5]]
    
    @test along_axis_free_monoid(; length = 5, winaxis = -2:0) == 
        [[1], [1, 2], [1, 2, 3], [2, 3, 4], [3, 4, 5]]
end

@testset "calc_fwd!" begin
    @test calc_fwd_free_monoid(;length=5, stride=1) == [[1], [2], [3], [4], [5]]
    @test calc_fwd_free_monoid(;length=5, stride=2) == [[1], [1,2], [3], [3,4], [5]]
    @test calc_fwd_free_monoid(;length=5, stride=3) == [[1], [1,2], [1,2,3], [4], [4,5]]
    @test calc_fwd_free_monoid(;length=5, stride=4) == [[1], [1,2], [1,2,3], [1,2,3,4], [5,]]
    @test calc_fwd_free_monoid(;length=5, stride=5) == [[1], [1,2], [1,2,3], [1,2,3,4], [1,2,3,4,5]]
    
end

@testset "calc_bwd!" begin
    @test calc_bwd_free_monoid(; length = 5, stride = 1) == [[1], [2], [3], [4], [5]]
    @test calc_bwd_free_monoid(; length = 5, stride = 2) == [[1, 2], [2], [3, 4], [4], [5]]
    @test calc_bwd_free_monoid(; length = 5, stride = 3) == [[1, 2, 3], [2, 3], [3], [4, 5], [5]]
    @test calc_bwd_free_monoid(; length = 5, stride = 4) == [[1, 2, 3, 4], [2, 3, 4], [3, 4], [4], [5]]
    @test calc_bwd_free_monoid(; length = 5, stride = 5) == [[1, 2, 3, 4, 5], [2, 3, 4, 5], [3, 4, 5], [4, 5], [5]] 
end


@testset "Digits" begin
    Digits = ReduceWindows.Digits
    for x in 1:260
        digits::Digits = @inferred Digits(x)
        @test Base.digits(x,base=2) == collect(digits)
        @test collect(digits) == [d for d in digits]
        n = length(collect(digits))
        @test collect(digits) == [Digits(x)[i] for i in 1:n]
    end

    @test length(Digits(0)) == 0
    @test length(Digits(1)) == 1
    @test length(Digits(2)) == 2
    @test length(Digits(3)) == 2
    @test length(Digits(4)) == 3
    @test length(Digits(5)) == 3
    @test length(Digits(6)) == 3
    @test length(Digits(7)) == 3
    @test length(Digits(8)) == 4
    @test length(Digits(9)) == 4
    @test length(Digits(10)) == 4
    @test length(Digits(11)) == 4
    @test length(Digits(12)) == 4
    @test length(Digits(13)) == 4
    @test length(Digits(14)) == 4
    @test length(Digits(15)) == 4
    @test length(Digits(16)) == 5
    @test length(Digits(17)) == 5

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
    rng = Xoshiro(1786867373)
    for _ in 1:1000
        lo = rand(rng, -100:0)
        hi = rand(rng, 0:100)
        win = (lo:hi,)
        len = rand(rng, 0:200)
        arr = randn(rng, len)
        @test reduce_window(+, arr, win) ≈ reduce_window_naive(+, arr, win)
    end
end

@testset "2d fuzz" begin
    rng = Xoshiro(2379087392)
    for _ in 1:1000
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
    rng = Xoshiro(1436607836)
    myadd(x,y) = x + y
    for _ in 1:100
        nd = rand(rng, 1:5)
        win = ntuple(nd) do _
            lo = rand(rng, -4:0)
            hi = rand(rng, 0:4)
            lo:hi
        end
        siz = Tuple(rand(rng, 0:8, nd))
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
