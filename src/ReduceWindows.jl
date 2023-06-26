module ReduceWindows
export reduce_window

using ArgCheck

struct OrderN end
struct OrderNLogK end
struct OrderNK end
struct MixedOrderN_OrderNLogK end

function fastmax(x,y)
    ifelse(x < y, y, x)
end
function fastmin(x,y)
    ifelse(x < y, x, y)
end

struct DeadPool{Tpl,Arr}
    template::Tpl
    dead::Vector{Arr}
    function DeadPool(template; ndead=0)
        @argcheck ndead >= 0
        dead = [similar(template) for _ in 1:ndead]
        Arr = eltype(dead)
        Tpl = typeof(template)
        new{Tpl, Arr}(template, dead)
    end
end
function alloc!(o::DeadPool)
    if isempty(o.dead)
        push!(o.dead, similar(o.template))
    end
    pop!(o.dead)
end
function free!(o::DeadPool, arr)
    @assert axes(arr) == axes(o.template)
    push!(o.dead, arr)
    return o
end

Base.@propagate_inbounds function apply_offset(I::CartesianIndex, ::Val{dim}, offset) where {dim}
    t = Tuple(I)
    t2 = Base.setindex(t, t[dim]+offset, dim)
    CartesianIndex(t2)
end

function floorlog2(x::Integer)
    @argcheck x > 0
    loglen = 0
    while (2^(loglen+1) <= x)
        loglen += 1
    end
    return loglen
end

function op_along_axis2!(f::F, out, arg2, Dim, offset, inds::CartesianIndices) where {F}
    @assert axes(out) == axes(arg2)
    @inbounds @simd for I1 in inds
        I2 = apply_offset(I1, Dim, offset)
        x1 = out[I1]
        x2 = arg2[I2]
        out[I1] = f(x1, x2)
    end
    return out
end

function copy_with_offset!(out, arg2, Dim, offset, inds)
    @assert axes(out) == axes(arg2)
    @inbounds @simd for I1 in inds
        I2 = apply_offset(I1, Dim, offset)
        x2 = arg2[I2]
        out[I1] = x2
    end
end

function first_inner_index_axis(outaxis::AbstractUnitRange, winaxis::AbstractUnitRange)
    first(outaxis) - first(winaxis)
end

function axes_unitrange(arr::AbstractArray{T,N})::NTuple{N,UnitRange{Int}} where {N,T}
    map(UnitRange{Int}, axes(arr))
end

unwrap(::Val{x}) where {x} = x
function along_axis_prefix!(f::F, out, inp, Dim::Val, winaxis)::typeof(out) where {F}
    dim = unwrap(Dim)
    # make sure the front elements of out along axis is correct
    lo = first(winaxis)
    hi = last(winaxis)
    if lo >= 0
        return out
    end
    inds = axes_unitrange(out)
    ifirst = firstindex(out, dim)
    ilast = lastindex(out, dim)
    istart = ifirst + 1
    istop = first_inner_index_axis(axes(out,dim), winaxis)-1
    istop = clamp(istop, ifirst, ilast)
    inds = Base.setindex(inds, istart:istop, dim)

    T = eltype(out)
    istop = lastindex(out, dim) - hi
    CI = CartesianIndices(inds)
    CI1, CI2 = split_indices_dim_istop(CI, dim, istop)
    @check length(CI1) + length(CI2) == length(CI)
    @inbounds @simd for I in CI1
        I1 = apply_offset(I, Dim, -1)
        x1 = out[I1]::T
        I2 = apply_offset(I, Dim, hi)
        x2 = inp[I2]
        out[I] = f(x1, x2)
    end
    @inbounds @simd for I in CI2
        I1 = apply_offset(I, Dim, -1)
        x1 = out[I1]::T
        out[I] = x1
    end
    return out
end

struct Digits
    x::Int
end
Base.eltype(::Type{Digits}) = Bool
function Base.iterate(d::Digits, state=d.x)
    if iszero(state)
        nothing
    else
        item = isodd(state)
        state = state >> 1
        (item, state)
    end
end
function Base.length(d::Digits)
    position_top_set_bit(d.x)
end
function Base.getindex(d::Digits, i::Int)::Bool
    isodd(d.x >> (i-1))
end
function position_top_set_bit(x::Integer)
    8*sizeof(x) - leading_zeros(x)
end

function split_indices_dim_istop(CI::CartesianIndices, dim, istop)
    inds = CI.indices
    ax = inds[dim]
    istop = min(istop, last(ax))
    ax1 = first(ax):istop
    istart = max(istop+1, first(ax))
    ax2 = istart:last(ax)
    @assert length(ax) == length(ax1) + length(ax2)
    inds1 = Base.setindex(inds, ax1, dim)
    inds2 = Base.setindex(inds, ax2, dim)
    CI1 = CartesianIndices(inds1)
    CI2 = CartesianIndices(inds2)
    return CI1, CI2
end

function split_indices_dim_offset(CI::CartesianIndices, dim, offset)
    inds = CI.indices
    ax = inds[dim]
    istop = last(ax)-offset
    split_indices_dim_istop(CI, dim, istop)
end

function power_stride!(f, out, inp, Dim::Val, offset)
    dim = unwrap(Dim)
    ci1, ci2 = split_indices_dim_offset(CartesianIndices(axes_unitrange(out)), dim, offset)
    @inbounds @simd for I in ci1
        I2 = apply_offset(I, Dim, offset)
        x1 = inp[I]
        x2 = inp[I2]
        out[I] = f(x1, x2)
    end
    @inbounds @simd for I in ci2
        x1 = inp[I]
        out[I] = x1
    end
    return out
end

@noinline function along_axis!(f::F, out, inp, Dim::Val, winaxis, alg::OrderNLogK, deadpool) where {F}
    dim = unwrap(Dim)
    lo = first(winaxis)
    hi = last(winaxis)
    digits_inner = Digits(length(winaxis))
    digits_first = Digits((lo >= 0) ? 0 : hi+1)
    offset_inner = lo 
    offset_first = 0
    winp = alloc!(deadpool)
    wout = alloc!(deadpool)
    copy!(wout, inp) # TODO deadpool can help elide copy
    out_inner_touched = false
    out_first_touched = false
    for iloglen in 1:64
        if 2^(iloglen) > 2*length(winaxis)
            break
        end
        arg2 = wout 
        if digits_inner[iloglen]
            istart = first_inner_index_axis(axes(out,dim), winaxis)
            istop = lastindex(out, dim)
            istop = min(istop, istop-offset_inner)
            inds = Base.setindex(axes_unitrange(out), istart:istop, dim)
            if out_inner_touched
                op_along_axis2!(f, out, arg2, Dim, offset_inner, CartesianIndices(inds))
            else
                copy_with_offset!(out, arg2, Dim, offset_inner, CartesianIndices(inds))
                out_inner_touched = true
            end
            offset_inner += 2^(iloglen-1)
        end
        if digits_first[iloglen]
            inds = axes_unitrange(out)
            i0 = first(inds[dim])
            inds = Base.setindex(inds, i0:i0, dim)
            if out_first_touched
                op_along_axis2!(f, out, arg2, Dim, offset_first, CartesianIndices(inds))
            else
                copy_with_offset!(out, arg2, Dim, offset_first, CartesianIndices(inds))
                out_first_touched = true
            end
            offset_first += 2^(iloglen-1)
        end
        winp, wout = wout, winp # TODO use deadpool instead of swapping
        power_stride!(f, wout, winp, Dim, 2^(iloglen-1))
    end
    free!(deadpool, wout)
    free!(deadpool, winp)
    # @assert out_first_touched
    @assert out_inner_touched
    along_axis_prefix!(f, out, inp, Dim, winaxis)
    return out
end

function shrink_window_axis(arraxis::AbstractUnitRange, winaxis::AbstractUnitRange)
    @argcheck !isempty(winaxis)
    @argcheck 0 in winaxis
    l = length(arraxis)
    istart = max(-l, first(winaxis))
    istop  = min(l, last(winaxis))
    return istart:istop
end

function resolve_window(array_axes, window)
    map(shrink_window_axis,array_axes, window)
end


"""

    reduce_window(f, arr, window)

Move a sliding window over the `arr` and apply `reduce(f, view(arr, shifted window...))`
For instance `window = (-1:2, 3:4)` will produce an output matrix with entries:

`out[i,j] = reduce(f, arr[(i-1):(i+2), (j+3):(j+4)])`

This is equation is true semantically, but in implementation much less work will be done.
Time complexity is `O(log(k) * n)` where 
* `n` is the size of the array: `n = length(arr)`
* `k` is the size of the window: `k = prod(length, window)`
Note `reduce_window` assumes, that `f` is associative and commutative.
"""
function reduce_window(f::F, arr::AbstractArray, window, alg=MixedOrderN_OrderNLogK(); deadpool=DeadPool(arr)) where {F}
    win = resolve_window(axes(arr), window)
    _reduce_window(f, arr, win, alg, deadpool)
end
function reduce_window(f::F, arr::AbstractArray, window, alg::OrderNK; deadpool=nothing) where {F}
    win = resolve_window(axes(arr), window)
    reduce_window_naive(f, arr, win)
end
function _reduce_window(f, arr, window, alg, deadpool)
    Arr = eltype(deadpool.dead)
    local inp::Arr
    if arr isa Arr
        inp = arr
    else
        inp = alloc!(deadpool)
        copy!(inp, arr)
    end
    for dim in 1:ndims(arr)
        Dim = Val(dim)
        winaxis = window[dim]
        if length(winaxis) > 1
            out = alloc!(deadpool)
            along_axis!(f, out, inp, Dim, winaxis, alg, deadpool)
            (inp === arr) || free!(deadpool, inp)
            inp = out
        else
            @assert winaxis == 0:0
        end
    end
    return inp
end

function reduce_window_naive(f, arr::AbstractArray{T,N}, window) where {T,N}
    win::NTuple{N,UnitRange} = resolve_window(axes(arr), window)
    out = similar(arr)
    for I in CartesianIndices(arr)
        inds::NTuple{N,UnitRange} = map(win, Tuple(I), axes(arr)) do r, i, ax
            istart = max(i + first(r), first(ax))
            istop = min(i+last(r), last(ax))
            istart:istop
        end
        patch = view(arr, inds...)
        val = reduce(f, patch)
        out[I] = val
    end
    return out
end

################################################################################
#### OrderN
################################################################################

function slices(arr::AbstractArray, Dim::Val{dim}, range) where {dim}
    axs = Base.setindex(axes(arr), range, dim)
    view(arr, axs...)
end

function calc_fwd_inner!(f::F, out, inp, Dim::Val{dim}, stride, starts::AbstractRange) where {F,dim}
    slices(out, Dim, starts) .= slices(inp, Dim, starts)
    for offset in 1:(stride-1)
        r1 = starts .+ offset
        r0 = starts .+ (offset -1)
        slices(out, Dim, r1) .= f.(slices(out, Dim, r0), slices(inp, Dim, r1))
    end
    return out
end

function calc_fwd_inner!(f::F, out, inp, Dim::Val{1}, stride, starts::AbstractRange) where {F}
    @inbounds for ci_rest in CartesianIndices(Base.tail(axes(inp)))
        i_rest = Tuple(ci_rest)
        for start in starts
            out[start, i_rest...] = inp[start, i_rest...]
            for offset in 1:(stride-1)
                i1 = start + offset
                i0 = start + (offset -1)
                out[i1, i_rest...] = f(out[i0, i_rest...], inp[i1, i_rest...])
            end
        end
    end
    return out
end

function calc_fwd!(f::F, out, inp, Dim::Val{dim}, stride) where {F,dim}
    @assert axes(out) == axes(inp)
    @assert 0 < stride <= size(out, dim)
    ifirst = firstindex(out, dim)
    ilast = lastindex(out, dim)
    starts = ifirst:stride:(ilast - (stride - 1))
    calc_fwd_inner!(f, out, inp, Dim, stride, starts)
    # boundary
    if ilast == last(starts) + (stride-1)
        return out
    end
    i0 = last(starts) + stride
    slices(out, Dim, i0) .= slices(inp, Dim, i0)
    for i in i0+1:ilast
        slices(out, Dim, i) .= f.(slices(out, Dim, i-1), slices(inp, Dim, i))
    end
    return out
end

function calc_bwd_inner!(f::F, out, inp, Dim::Val{dim}, stride, starts::AbstractRange) where {F,dim}
    slices(out, Dim, starts) .= slices(inp, Dim, starts)
    for offset in 1:(stride-1)
        r1 = starts .- offset
        r0 = starts .- (offset -1)
        slices(out, Dim, r1) .= f.(slices(inp, Dim, r1), slices(out, Dim, r0), )
    end
    return out
end

function calc_bwd_inner!(f::F, out, inp, Dim::Val{1}, stride, starts::AbstractRange) where {F}
    @inbounds for ci_rest in CartesianIndices(Base.tail(axes(inp)))
        i_rest = Tuple(ci_rest)
        for start in starts
            out[start, i_rest...] = inp[start, i_rest...]
            for offset in 1:(stride-1)
                i1 = start - offset
                i0 = start - (offset -1)
                out[i1, i_rest...] = f(inp[i1, i_rest...], out[i0, i_rest...])
            end
        end
    end
    return out
end

function calc_bwd!(f::F, out, inp, Dim::Val{dim}, stride) where {F,dim}
    @assert axes(out) == axes(inp)
    @assert 0 < stride <= size(out, dim)
    ifirst = firstindex(out, dim)
    ilast = lastindex(out, dim)
    starts = ifirst+(stride-1):stride:ilast
    calc_bwd_inner!(f, out, inp, Dim, stride, starts)
    # boundary
    if ilast == last(starts)
        return out
    end
    slices(out, Dim, ilast) .= slices(inp, Dim, ilast)
    for i in (ilast-1):(-1):(last(starts)+1)
        slices(out, Dim, i) .= f.(slices(inp, Dim, i), slices(out, Dim, i+1))
    end
    return out
end

@noinline function along_axis!(f::F, out::AbstractArray, inp::AbstractArray, 
        Dim::Val{1}, winaxis::AbstractUnitRange, ::MixedOrderN_OrderNLogK, deadpool::DeadPool) where {F}
    along_axis!(f, out, inp, Dim, winaxis, OrderNLogK(), deadpool)
end
@noinline function along_axis!(f::F, out::AbstractArray, inp::AbstractArray, 
        Dim::Val{dim}, winaxis::AbstractUnitRange, ::MixedOrderN_OrderNLogK, deadpool::DeadPool) where {F, dim}
    along_axis!(f, out, inp, Dim, winaxis, OrderN(), deadpool)
end

@noinline function along_axis!(f::F, out::AbstractArray, inp::AbstractArray, 
        Dim::Val{dim}, winaxis::AbstractUnitRange, ::OrderN, deadpool::DeadPool) where {F,dim}
    @assert axes(out) == axes(inp)
    @assert 0 in winaxis
    fwd = alloc!(deadpool)
    bwd = alloc!(deadpool)
    lo = first(winaxis)
    hi = last(winaxis)
    stride = hi - lo
    if stride > size(out, dim)
        # TODO handle this case in O(N)
        return along_axis!(f, out, inp,
            Dim, winaxis, OrderNLogK(), deadpool)
    end
    @argcheck 0 < stride <= size(out, dim)
    calc_fwd!(f, fwd, inp, Dim, stride)
    calc_bwd!(f, bwd, inp, Dim, stride)

    to_fwd = hi
    to_bwd = lo
    ifirst = firstindex(out, dim)
    ilast = lastindex(out, dim)
    # boundary pre
    r_pre = ifirst:ifirst-lo-1
    slices(out, Dim, r_pre) .= slices(fwd, Dim, r_pre .+ to_fwd)
    # boundary post
    ilast_fwd = last(range(ifirst, step=stride, stop=ilast)) + stride - 1
    for i in (ilast-hi+1):ilast
        if i + to_fwd > ilast_fwd
            slices(out, Dim, i) .= slices(bwd, Dim, i + to_bwd)
        else
            slices(out, Dim, i) .= f.(slices(bwd, Dim, i + to_bwd),
                                       slices(fwd, Dim, ilast,))
        end
    end
    # inner
    r_inner = (ifirst-to_bwd):(ilast-to_fwd)
    r_fwd = r_inner .+ to_fwd
    r_bwd = r_inner .+ to_bwd
    slices(out, Dim, r_inner) .= f.(slices(bwd, Dim, r_bwd), slices(fwd, Dim, r_fwd))
    free!(deadpool, fwd)
    free!(deadpool, bwd)
    return out
end


end
