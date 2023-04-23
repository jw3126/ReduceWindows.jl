using ReduceWindows: reduce_window, reduce_window_naive
using Test
using Random: Xoshiro

@testset "1d explicit" begin
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
