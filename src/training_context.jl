"""
$TYPEDEF

Lightweight mutable context object passed to metrics during training.

# Fields
$TYPEDFIELDS

# Notes
- `policy`, `maximizer_kwargs`, and `other_fields` are constant after construction; only `epoch` is intended to be mutated.
"""
mutable struct TrainingContext{P,F,O<:NamedTuple}
    "the DFLPolicy being trained"
    const policy::P
    "current epoch number (mutated in-place during training)"
    epoch::Int
    "function to extract keyword arguments for maximizer calls from data samples"
    const maximizer_kwargs::F
    "`NamedTuple` container of additional algorithm-specific configuration (e.g., loss)"
    const other_fields::O
end

function TrainingContext(; policy, epoch, maximizer_kwargs=get_info, kwargs...)
    other_fields = isempty(kwargs) ? NamedTuple() : NamedTuple(kwargs)
    return TrainingContext(policy, epoch, maximizer_kwargs, other_fields)
end

function Base.show(io::IO, ctx::TrainingContext)
    print(io, "TrainingContext(")
    print(io, "epoch=$(ctx.epoch), ")
    print(io, "policy=$(typeof(ctx.policy))")
    if !isempty(ctx.other_fields)
        print(io, ", other_fields=$(keys(ctx.other_fields))")
    end
    return print(io, ")")
end

function Base.hasproperty(ctx::TrainingContext, name::Symbol)
    return name in fieldnames(TrainingContext) ||
           (!isempty(ctx.other_fields) && haskey(ctx.other_fields, name))
end

# Support for haskey to maintain compatibility with NamedTuple-style access
Base.haskey(ctx::TrainingContext, key::Symbol) = hasproperty(ctx, key)

# Property access for additional fields stored in other_fields
function Base.getproperty(ctx::TrainingContext, name::Symbol)
    if name in fieldnames(TrainingContext)
        return getfield(ctx, name)
    elseif !isempty(ctx.other_fields) && haskey(ctx.other_fields, name)
        return ctx.other_fields[name]
    else
        throw(ArgumentError("TrainingContext $ctx has no field $name"))
    end
end

"""
$TYPEDSIGNATURES

Advance the epoch counter in the training context by one.
"""
function next_epoch!(ctx::TrainingContext)
    ctx.epoch += 1
    return nothing
end
