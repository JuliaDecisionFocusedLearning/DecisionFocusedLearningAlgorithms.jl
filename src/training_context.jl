"""
$TYPEDEF

Lightweight mutable context object passed to metrics during training.

# Fields
$TYPEDFIELDS

# Notes
- `model`, `maximizer`, and `other_fields` are constant after construction; only `epoch` is intended to be mutated.
"""
mutable struct TrainingContext{M,MX,O<:NamedTuple}
    "the ML model being trained"
    const model::M
    "current epoch number (mutated in-place during training)"
    epoch::Int
    "CO maximizer used for decision-making (can be any callable)"
    const maximizer::MX
    "`NamedTuple` container of optional algorithm-specific values"
    const other_fields::O
end

function TrainingContext(; model, epoch, maximizer, kwargs...)
    other_fields = isempty(kwargs) ? NamedTuple() : NamedTuple(kwargs)
    return TrainingContext(model, epoch, maximizer, other_fields)
end

function Base.show(io::IO, ctx::TrainingContext)
    print(io, "TrainingContext(")
    print(io, "epoch=$(ctx.epoch), ")
    print(io, "model=$(typeof(ctx.model))")
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
