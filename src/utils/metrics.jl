# TODO: review and tests

# Helper functions for nested callbacks
function _flatten_callbacks(callbacks::NamedTuple, prefix="")
    result = NamedTuple()
    for (key, value) in pairs(callbacks)
        new_key = isempty(prefix) ? key : Symbol("$(prefix)_$(key)")
        if isa(value, NamedTuple)
            result = merge(result, _flatten_callbacks(value, string(new_key)))
        else
            result = merge(result, NamedTuple{(new_key,)}((value,)))
        end
    end
    return result
end

function _unflatten_metrics(flat_metrics::NamedTuple, original_structure::NamedTuple)
    if isempty(original_structure)
        return NamedTuple()
    end

    result = NamedTuple()
    for (key, value) in pairs(original_structure)
        if isa(value, NamedTuple)
            # Recursively unflatten nested structure
            nested_result = _unflatten_metrics(flat_metrics, value)
            result = merge(result, NamedTuple{(key,)}((nested_result,)))
        else
            # This is a leaf callback, get its metric
            result = merge(result, NamedTuple{(key,)}((flat_metrics[key],)))
        end
    end
    return result
end

function _initialize_nested_metrics(callbacks::NamedTuple, model, maximizer, epoch)
    if isempty(callbacks)
        return NamedTuple()
    end

    result = NamedTuple()
    for (key, value) in pairs(callbacks)
        if isa(value, NamedTuple)
            # Recursively handle nested callbacks
            nested_metrics = _initialize_nested_metrics(value, model, maximizer, epoch)
            result = merge(result, NamedTuple{(key,)}((nested_metrics,)))
        else
            # This is a leaf callback
            initial_value = try
                value(model, maximizer, epoch)
            catch e
                @warn "Metrics callback $key failed at initialization" exception = e
                nothing
            end
            result = merge(result, NamedTuple{(key,)}(([initial_value],)))
        end
    end
    return result
end

function _call_nested_callbacks(callbacks::NamedTuple, model, maximizer, epoch)
    if isempty(callbacks)
        return NamedTuple()
    end

    result = NamedTuple()
    for (key, value) in pairs(callbacks)
        if isa(value, NamedTuple)
            # Recursively handle nested callbacks
            nested_metrics = _call_nested_callbacks(value, model, maximizer, epoch)
            result = merge(result, NamedTuple{(key,)}((nested_metrics,)))
        else
            # This is a leaf callback
            metric_value = try
                value(model, maximizer, epoch)
            catch e
                @warn "Metrics callback $key failed" exception = e
                nothing
            end
            result = merge(result, NamedTuple{(key,)}((metric_value,)))
        end
    end
    return result
end

function _push_nested_metrics!(metrics_history, epoch_metrics)
    for (key, value) in pairs(epoch_metrics)
        if isa(value, NamedTuple)
            # Recursively handle nested metrics
            _push_nested_metrics!(metrics_history[key], value)
        else
            # This is a leaf metric
            push!(metrics_history[key], value)
        end
    end
end

# Helper function to flatten metrics across DAgger iterations
function _flatten_dagger_metrics(all_metrics)
    if isempty(all_metrics)
        return NamedTuple()
    end

    # Get the structure from the first iteration
    first_metrics = all_metrics[1]
    flattened = NamedTuple()

    for (key, _) in pairs(first_metrics)
        # For first iteration: keep all values
        # For subsequent iterations: skip the first epoch (index 1)
        all_values = vcat(
            [
                iter == 1 ? metrics[key] : metrics[key][2:end] for
                (iter, metrics) in enumerate(all_metrics)
            ]...,
        )
        flattened = merge(flattened, NamedTuple{(key,)}((all_values,)))
    end

    return flattened
end
