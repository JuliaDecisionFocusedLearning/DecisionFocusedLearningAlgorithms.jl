"""
$TYPEDEF

# Fields
$TYPEDFIELDS
"""
struct TrainingContext{M,O}
    "ML model"
    model::M
    "Current epoch number"
    epoch::Int
    "CO Maximizer function"
    maximizer::Function
    "Additional fields"
    other_fields::O
end

function TrainingContext(
    model,
    epoch,
    maximizer;
    kwargs...,
)
    other_fields = isempty(kwargs) ? NamedTuple() : NamedTuple(kwargs)
    return TrainingContext(
        model,
        epoch,
        maximizer,
        other_fields,
    )
end

# Convenience constructor that matches the old NamedTuple interface
function TrainingContext(;
    model,
    epoch,
    maximizer,
    kwargs...,
)
    other_fields = isempty(kwargs) ? NamedTuple() : NamedTuple(kwargs)
    return TrainingContext(
        model,
        epoch,
        maximizer,
        other_fields,
    )
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

# Support for iteration over context properties (useful for debugging)
function Base.propertynames(ctx::TrainingContext)
    return (fieldnames(TrainingContext)..., keys(ctx.other_fields)...)
end

# Helper method to create a new context with updated fields
function update_context(ctx::TrainingContext; kwargs...)
    # Extract all current field values
    new_model = get(kwargs, :model, ctx.model)
    new_epoch = get(kwargs, :epoch, ctx.epoch)
    new_maximizer = get(kwargs, :maximizer, ctx.maximizer)
    new_train_dataset = get(kwargs, :train_dataset, ctx.train_dataset)
    new_validation_dataset = get(kwargs, :validation_dataset, ctx.validation_dataset)
    # new_train_loss = get(kwargs, :train_loss, ctx.train_loss)
    # new_val_loss = get(kwargs, :val_loss, ctx.val_loss)

    # Merge other_fields with new kwargs
    new_other_fields = merge(
        ctx.other_fields,
        filter(
            kv ->
                kv.first ∉ (
                    :model,
                    :epoch,
                    :maximizer,
                    :train_dataset,
                    :validation_dataset,
                    # :train_loss,
                    # :val_loss,
                ),
            kwargs,
        ),
    )

    return TrainingContext(
        new_model,
        new_epoch,
        new_maximizer,
        new_train_dataset,
        new_validation_dataset,
        # new_train_loss,
        # new_val_loss,
        new_other_fields,
    )
end
