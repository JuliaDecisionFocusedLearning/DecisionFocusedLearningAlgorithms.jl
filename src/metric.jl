abstract type AbstractMetric end

function reset!(metric::AbstractMetric) end
function update!(metric::AbstractMetric; kwargs...) end
function evaluate!(metric::AbstractMetric, policy, dataset; kwargs...) end
