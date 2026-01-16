"""
$TYPEDEF

An abstract type for decision-focused learning algorithms.
"""
abstract type AbstractAlgorithm end

"""
$TYPEDEF

An abstract type for imitation learning algorithms.

All subtypes must implement:
- `train_policy!(algorithm::AbstractImitationAlgorithm, model, maximizer, train_data; epochs, metrics)`
"""
abstract type AbstractImitationAlgorithm <: AbstractAlgorithm end
