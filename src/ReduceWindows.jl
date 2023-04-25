module ReduceWindows
export reduce_window

using ArgCheck

function fastmax(x,y)
    ifelse(x < y, y, x)
end
function fastmin(x,y)
    ifelse(x < y, x, y)
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
function add_along_axis_prefix!(f::F, out, inp, Dim::Val, window)::typeof(out) where {F}
    dim = unwrap(Dim)
    # make sure the front elements of out along axis is correct
    winaxis = window[dim]
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
    8*sizeof(d.x)
end
function Base.getindex(d::Digits, i::Int)::Bool
    isodd(d.x >> (i-1))
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

@noinline function add_along_axis!(f::F, out, inp, Dim::Val, window, workspace) where {F}
    dim = unwrap(Dim)
    winaxis = window[dim]
    lo = first(winaxis)
    hi = last(winaxis)
    digits_inner = Digits(length(winaxis))
    digits_first = Digits((lo >= 0) ? 0 : hi+1)
    offset_inner = lo 
    offset_first = 0
    (;winp, wout) = workspace
    copy!(wout, inp)
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
        winp, wout = wout, winp
        power_stride!(f, wout, winp, Dim, 2^(iloglen-1))
    end
    # @assert out_first_touched
    @assert out_inner_touched
    add_along_axis_prefix!(f, out, inp, Dim, window)
    return out
end

function shrink_window_axis(arraxis::AbstractUnitRange, windowaxis::AbstractUnitRange)
    @argcheck !isempty(windowaxis)
    @argcheck 0 in windowaxis
    l = length(arraxis)
    istart = max(-l, first(windowaxis))
    istop  = min(l, last(windowaxis))
    return istart:istop
end

function resolve_window(array_axes, window)
    map(shrink_window_axis,array_axes, window)
end

"""

    reduce_window(f, arr, window)

Move a sliding window over the `arr` and apply `reduce(f, view(arr, window...))`
This function assumes, that `f` is associative, commutative.

Time complexity is `O(length(arr) * log(prod(length,window)))`.
"""
function reduce_window(f::F, arr, window) where {F}
    win = resolve_window(axes(arr), window)
    workspace = (;winp=similar(arr), wout=similar(arr))
    out = similar(arr)
    inp = copy!(similar(arr), arr)
    for dim in 1:ndims(arr)
        Dim = Val(dim)
        add_along_axis!(f, out, inp, Dim, win, workspace)
        (inp, out) = (out, inp)
    end
    (inp, out) = (out, inp)
    return out
end

function reduce_window_naive(f, arr::AbstractArray{T,N}, window) where {T,N}
    win::NTuple{N,UnitRange} = resolve_window(axes(arr), window)
    out = similar(arr)
    for I in CartesianIndices(arr)
        inds = map(win, Tuple(I), axes(arr)) do r, i, ax
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

end
