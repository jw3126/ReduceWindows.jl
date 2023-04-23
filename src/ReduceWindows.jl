module ReduceWindows

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

function populate_workspace_along_axis!(f, arr, dim, window, neutral_element, workspace_vector)
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

function op_along_axis2(f, out, arg2, dim, offset, inds::CartesianIndices)
    @assert axes(out) == axes(arg2)
    # TODO end
    for I1 in inds
        I2 = apply_offset(I1, dim, offset)
        x1 = out[I1]
        x2 = arg2[I2]
        out[I1] = f(x1, x2)
    end
    return out
end

function add_along_axis_first!(f, out, dim, window, neutral_element, workspace_vector)
    # make sure the first element of out along axis is correct
    winaxis = window[dim]
    lo = first(winaxis)
    @assert lo <= 0
    inds = axes(out)
    i0 = first(inds[dim])
    inds = Base.setindex(inds, i0:i0, dim)
    hi = last(winaxis)
    @assert hi >= 0
    digits = Base.digits(hi+1, base=2)
    offset = 0
    for (iloglen,dig) in enumerate(digits)
        @assert dig in 0:1
        if Bool(dig)
            arg2 = workspace_vector[iloglen]
            op_along_axis2(f, out, arg2, dim, offset, CartesianIndices(inds))
            offset += 2^(iloglen-1)
        end
    end
    return out
end

function first_inner_index_axis(outaxis::AbstractUnitRange, winaxis::AbstractUnitRange)
    first(outaxis) - first(winaxis)
end

function add_along_axis_prefix!(f, out, dim, window, neutral_element, workspace_vector)
    # make sure the front elements of out along axis is correct
    winaxis = window[dim]
    lo = first(winaxis)
    hi = last(winaxis)
    if lo >= 0
        return out
    end
    add_along_axis_first!(f, out, dim, window, neutral_element, workspace_vector)
    inds = axes(out)
    ifirst = firstindex(out, dim)
    ilast = lastindex(out, dim)
    istart = min(ifirst + 1, ilast)
    istop = first_inner_index_axis(axes(out,dim), winaxis)-1
    istop = clamp(istop, ifirst, ilast)
    inds = Base.setindex(inds, istart:istop, dim)
    inp = first(workspace_vector)

    for I in CartesianIndices(inds)
        I1 = apply_offset(I, dim, -1)
        x1 = out[I1]
        I2 = apply_offset(I, dim, hi)
        x2 = get(inp, I2, neutral_element) # TODO SIMD
        out[I] = f(x1, x2)
    end
    return out
end

function add_along_axis!(f, out, dim, window, neutral_element, workspace_vector)
    add_along_axis_prefix!(f, out, dim, window, neutral_element, workspace_vector)
    winaxis = window[dim]
    istart = first_inner_index_axis(axes(out,dim), winaxis)
    digits = Base.digits(length(winaxis), base=2)
    offset = first(winaxis)
    for (iloglen,dig) in enumerate(digits)
        @assert dig in 0:1
        if Bool(dig)
            istop = lastindex(out, dim)
            istop = min(istop, istop-offset)
            inds = Base.setindex(axes(out), istart:istop, dim)
            arg2 = workspace_vector[iloglen]
            op_along_axis2(f, out, arg2, dim, offset, CartesianIndices(inds))
            offset += 2^(iloglen-1)
        end
    end
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

function reduce_window(f, arr, window; neutral_element=get_neutral_element(f, eltype(arr)))
    @argcheck ndims(arr) == 1
    window = resolve_window(axes(arr), window)
    workspace_vector = alloc_workspace_vector(arr, window)
    out = fill!(similar(arr), neutral_element)
    for dim in 1:ndims(arr)
        populate_workspace_along_axis!(f, arr, dim, window, neutral_element, workspace_vector)
        add_along_axis!(f, out, dim, window, neutral_element, workspace_vector)
    end
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
