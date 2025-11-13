using JLD2
using Flux
using DecisionFocusedLearningBenchmarks
const DVSP = DecisionFocusedLearningBenchmarks.DynamicVehicleScheduling
using ValueHistories
using Plots

b = DynamicVehicleSchedulingBenchmark(; max_requests_per_epoch=50)

logs = JLD2.load(joinpath(@__DIR__, "logs.jld2"))
model = logs["model"]
history = logs["history"]

epochs, train_losses = get(history, :training_loss)
epochs, val_losses = get(history, :validation_loss)
epochs, train_obj = get(history, :train_obj)
epochs, val_obj = get(history, :val_obj)

slice = 1:25#length(epochs)
loss_fig = plot(
    epochs[slice], train_losses[slice]; label="Train Loss", xlabel="Epoch", ylabel="Loss"
)
plot!(loss_fig, epochs[slice], val_losses[slice]; label="Val Loss")

cost_fig = plot(
    epochs[slice], -train_obj[slice]; label="Train cost", xlabel="Epoch", ylabel="Cost"
)
plot!(cost_fig, epochs[slice], -val_obj[slice]; label="Val cost")

data = JLD2.load(joinpath(@__DIR__, "saved_data.jld2"))
instances = data["instances"]
dataset = data["dataset"]

extrema(dataset[1].info.static_instance.duration)

nb_instances = length(dataset)
for instance_id in 1:nb_instances
    dataset[instance_id].info.static_instance.duration .=
        instances[instance_id].duration ./ 1000
end

extrema(dataset[1].info.static_instance.duration)

dataset[1].info
old_instance = dataset[1].info
(;
    epoch_duration,
    last_epoch,
    max_requests_per_epoch,
    Δ_dispatch,
    static_instance,
    two_dimensional_features,
) = old_instance
instance = DVSP.Instance(
    static_instance;
    epoch_duration,
    two_dimensional_features,
    Δ_dispatch,
    max_requests_per_epoch=50,
)

environments = generate_environments(b, [DataSample(; info=instance)])
env = first(environments)

policies = generate_policies(b)
lazy = policies[1]
greedy = policies[2]

greedy_cost, greedy_data = evaluate_policy!(greedy, first(environments))
lazy_cost, lazy_data = evaluate_policy!(lazy, first(environments))
anticipative_cost, anticipative_data = generate_anticipative_solution(
    b, first(environments); reset_env=true
)
greedy_cost
lazy_cost
anticipative_cost

struct DFLPolicy{F,M}
    model::F
    maximizer::M
end

function (p::DFLPolicy)(env)
    x, state = observe(env)
    θ = p.model(x)
    y = p.maximizer(θ; instance=state)
    return DVSP.decode_bitmatrix_to_routes(y)
end

maximizer = generate_maximizer(b)
policy = Policy("", "", DFLPolicy(model, maximizer))

dfl_cost, dfl_data = evaluate_policy!(policy, first(environments))

using JSON3
open("greedy.json", "w") do f
    JSON3.pretty(f, JSON3.write(DVSP.build_plot_data(greedy_data)))
    println(f)
end
open("lazy.json", "w") do f
    JSON3.pretty(f, JSON3.write(DVSP.build_plot_data(lazy_data)))
    println(f)
end
open("dfl.json", "w") do f
    JSON3.pretty(f, JSON3.write(DVSP.build_plot_data(dfl_data)))
    println(f)
end
open("anticipative.json", "w") do f
    JSON3.pretty(f, JSON3.write(DVSP.build_plot_data(anticipative_data)))
    println(f)
end
