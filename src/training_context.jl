"""
    TrainingContext{M,O}

Lightweight mutable context object passed to metrics during training.

Fields
- `model::M`: The ML model being trained
- `epoch::Int`: Current epoch number (mutated in-place during training)
- `maximizer`: CO maximizer used for decision-making (can be any callable)
- `other_fields::O`: NamedTuple of optional algorithm-specific values

Notes
- `model`, `maximizer`, and `other_fields` are constant after construction; only `epoch` is intended to be mutated.
- Use `update_context` to obtain a shallow copy with updated `other_fields` when needed.
"""
mutable struct TrainingContext{M,MX,O}
    "ML model"
    const model::M
    "Current epoch number"
    epoch::Int
    "CO Maximizer (any callable)"
    const maximizer::MX
    "Additional fields"
    const other_fields::O
end

function TrainingContext(model, epoch, maximizer; kwargs...)
    other_fields = isempty(kwargs) ? NamedTuple() : NamedTuple(kwargs)
    return TrainingContext(model, epoch, maximizer, other_fields)
end

function TrainingContext(; model, epoch, maximizer, kwargs...)
    return TrainingContext(model, epoch, maximizer; kwargs...)
end

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

function Base.hasproperty(ctx::TrainingContext, name::Symbol)
    return name in fieldnames(TrainingContext) ||
           (!isempty(ctx.other_fields) && haskey(ctx.other_fields, name))
end

# Support for haskey to maintain compatibility with NamedTuple-style access
Base.haskey(ctx::TrainingContext, key::Symbol) = hasproperty(ctx, key)

# Pretty printing for TrainingContext
function Base.show(io::IO, ctx::TrainingContext)
    print(io, "TrainingContext(")
    print(io, "epoch=$(ctx.epoch), ")
    print(io, "model=$(typeof(ctx.model))")
    if !isempty(ctx.other_fields)
        print(io, ", other_fields=$(keys(ctx.other_fields))")
    end
    return print(io, ")")
end

# # Helper to return a shallow copy with updated additional fields
# function update_context(ctx::TrainingContext; kwargs...)
#     new_model = get(kwargs, :model, ctx.model)
#     new_epoch = get(kwargs, :epoch, ctx.epoch)
#     new_maximizer = get(kwargs, :maximizer, ctx.maximizer)

#     # Merge other_fields with new kwargs, excluding core fields
#     new_other_fields = merge(
#         ctx.other_fields, filter(kv -> kv.first ∉ (:model, :epoch, :maximizer), kwargs)
#     )

#     return TrainingContext(new_model, new_epoch, new_maximizer, new_other_fields)
# end
