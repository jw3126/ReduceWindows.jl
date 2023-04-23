module ReduceWindows
export reduce_window

using ArgCheck

function fastmax(x,y)
    ifelse(x < y, y, x)
end
function fastmin(x,y)
    ifelse(x < y, x, y)
end
function apply_offset(I::CartesianIndex, dim, offset)
    t = Tuple(I)
    t2 = Base.setindex(t, t[dim]+offset, dim)
    CartesianIndex(t2)
end

function get_workspace_loglen(workspace_vector::AbstractVector, loglen)
    Base.require_one_based_indexing(workspace_vector)
    workspace_vector[loglen+1]
end

function alloc_workspace_vector(arr, window)
    @argcheck ndims(arr) == length(window)
    loglen = floorlog2(maximum(length, window))
    map(1:loglen+1) do _
        similar(arr)
    end
end

function floorlog2(x::Integer)
    @argcheck x > 0
    loglen = 0
    while (2^(loglen+1) <= x)
        loglen += 1
    end
    return loglen
end

@noinline function populate_workspace_along_axis!(f::F, arr, dim, window, neutral_element, workspace_vector) where {F}
    if length(window[dim]) == 0
        return workspace_vector
    end
    required_loglength = floorlog2(length(window[dim]))
    loglen = 0
    table = get_workspace_loglen(workspace_vector, loglen)
    copy!(table, arr)
    table_prev = table
    while loglen < required_loglength
        table_prev = get_workspace_loglen(workspace_vector, loglen)
        table_next = get_workspace_loglen(workspace_vector, loglen+1)
        offset = 2^loglen
        for I in CartesianIndices(table_prev)
            I2 = apply_offset(I, dim, offset)
            x1 = table_prev[I]
            x2 = get(table_prev, I2, neutral_element) # TODO SIMD friendly
            table_next[I] = f(x1, x2)
        end
        loglen += 1
    end
    return workspace_vector
end

function op_along_axis2!(f::F, out, arg2, dim, offset, inds::CartesianIndices) where {F}
    @assert axes(out) == axes(arg2)
    for I1 in inds
        I2 = apply_offset(I1, dim, offset)
        x1 = out[I1]
        x2 = arg2[I2]
        out[I1] = f(x1, x2)
    end
    return out
end

function first_inner_index_axis(outaxis::AbstractUnitRange, winaxis::AbstractUnitRange)
    first(outaxis) - first(winaxis)
end

function axes_unitrange(arr::AbstractArray{T,N})::NTuple{N,UnitRange{Int}} where {N,T}
    map(UnitRange{Int}, axes(arr))
end

function add_along_axis_prefix!(f::F, out, dim, window, neutral_element, workspace_vector)::typeof(out) where {F}
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
    inp = first(workspace_vector)

    T = eltype(out)
    for I in CartesianIndices(inds)
        I1 = apply_offset(I, dim, -1)
        x1 = out[I1]::T
        I2 = apply_offset(I, dim, hi)
        x2 = get(inp, I2, neutral_element)::T # TODO SIMD
        out[I] = f(x1, x2)
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

@noinline function add_along_axis!(f::F, out, dim, window, neutral_element, workspace_vector) where {F}
    winaxis = window[dim]
    lo = first(winaxis)
    hi = last(winaxis)
    digits_inner = Digits(length(winaxis))
    digits_first = Digits((lo >= 0) ? 0 : hi+1)
    offset_inner = lo 
    offset_first = 0
    for iloglen in 1:32
        if digits_inner[iloglen]
            istart = first_inner_index_axis(axes(out,dim), winaxis)
            istop = lastindex(out, dim)
            istop = min(istop, istop-offset_inner)
            inds = Base.setindex(axes_unitrange(out), istart:istop, dim)
            arg2 = workspace_vector[iloglen]
            op_along_axis2!(f, out, arg2, dim, offset_inner, CartesianIndices(inds))
            offset_inner += 2^(iloglen-1)
        end
        if digits_first[iloglen]
            inds = axes_unitrange(out)
            i0 = first(inds[dim])
            inds = Base.setindex(inds, i0:i0, dim)
            arg2 = workspace_vector[iloglen]
            op_along_axis2!(f, out, arg2, dim, offset_first, CartesianIndices(inds))
            offset_first += 2^(iloglen-1)
        end
        if 2^(iloglen-1) > size(out, dim)
            break
        end
    end
    add_along_axis_prefix!(f, out, dim, window, neutral_element, workspace_vector)
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

function reduce_window(f::F, arr, window; neutral_element=get_neutral_element(f, eltype(arr))) where {F}
    win = resolve_window(axes(arr), window)
    workspace_vector = alloc_workspace_vector(arr, win)
    out = similar(arr)
    inp = copy!(similar(arr), arr)
    for dim in 1:ndims(arr)
        populate_workspace_along_axis!(f, inp, dim, win, neutral_element, workspace_vector)
        fill!(out, neutral_element)
        add_along_axis!(f, out, dim, win, neutral_element, workspace_vector)
        (inp, out) = (out, inp)
    end
    (inp, out) = (out, inp)
    return out
end

function reduce_window_naive(f, arr::AbstractArray{T,N}, window; 
        neutral_element=get_neutral_element(f,T)) where {T,N}
    win::NTuple{N,UnitRange} = resolve_window(axes(arr), window)
    out = similar(arr)
    for I in CartesianIndices(arr)
        inds = map(win, Tuple(I), axes(arr)) do r, i, ax
            istart = max(i + first(r), first(ax))
            istop = min(i+last(r), last(ax))
            istart:istop
        end
        patch = view(arr, inds...)
        val = reduce(f, patch, init=neutral_element)
        out[I] = val
    end
    return out
end

function get_neutral_element(::typeof(min), T)
    typemax(T)
end
function get_neutral_element(::typeof(max), T)
    typemin(T)
end
function get_neutral_element(::typeof(fastmin), T)
    typemax(T)
end
function get_neutral_element(::typeof(fastmax), T)
    typemin(T)
end
function get_neutral_element(::typeof(+), T)
    zero(T)
end






# Write your package code here.

end
