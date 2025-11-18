using DecisionFocusedLearningAlgorithms
using DecisionFocusedLearningBenchmarks
using MLUtils: splitobs
using ValueHistories
using Plots
using Random
using Statistics
using JLD2
using Flux
const DVSP = DecisionFocusedLearningBenchmarks.DynamicVehicleScheduling

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

b = DynamicVehicleSchedulingBenchmark(; max_requests_per_epoch=10)

dataset = generate_dataset(b, 100)
train_instances, validation_instances, test_instances = splitobs(dataset; at=(0.3, 0.3))
train_environments = generate_environments(b, train_instances)
validation_environments = generate_environments(b, validation_instances)
test_environments = generate_environments(b, test_instances)

observe(first(train_environments))[1]

train_dataset = vcat(map(train_environments) do env
    v, y = generate_anticipative_solution(b, env; reset_env=true)
    return y
end...)

val_dataset = vcat(map(validation_environments) do env
    v, y = generate_anticipative_solution(b, env; reset_env=true)
    return y
end...)

shuffle!(train_dataset)
shuffle!(val_dataset)

initial_model = generate_statistical_model(b; seed=0)
Random.seed!(42)
initial_model = Chain(
    Dense(27 => 10, relu), Dense(10 => 10, relu), Dense(10 => 10, relu), Dense(10 => 1), vec
)
maximizer = generate_maximizer(b)

model = deepcopy(initial_model)
callbacks = [
    Metric(
        :train_obj,
        (data, ctx) -> mean(
            evaluate_policy!(Policy("", "", DFLPolicy(ctx.model, ctx.maximizer)), data)[1],
        );
        on=train_environments,
    ),
    Metric(
        :val_obj,
        (data, ctx) -> mean(
            evaluate_policy!(Policy("", "", DFLPolicy(ctx.model, ctx.maximizer)), data)[1],
        );
        on=validation_environments,
    ),
];
typeof(callbacks)

history = fyl_train_model!(
    model,
    maximizer,
    train_dataset,
    val_dataset;
    epochs=25,
    maximizer_kwargs=(sample -> (; instance=sample.info.state)),
    callbacks=callbacks,
)

# JLD2.jldsave(joinpath(@__DIR__, "logs_2.jld2"); model=model, history=history)

epochs, train_losses = get(history, :training_loss)
epochs, val_losses = get(history, :validation_loss)
epochs, train_obj = get(history, :train_obj)
epochs, val_obj = get(history, :val_obj)

slice = 1:length(epochs)
loss_fig = plot(
    epochs[slice], train_losses[slice]; label="Train Loss", xlabel="Epoch", ylabel="Loss"
)
plot!(loss_fig, epochs[slice], val_losses[slice]; label="Val Loss")
savefig(loss_fig, "dfl_policy_loss.png")

cost_fig = plot(
    epochs[slice], -train_obj[slice]; label="Train cost", xlabel="Epoch", ylabel="Cost"
)
plot!(cost_fig, epochs[slice], -val_obj[slice]; label="Val cost")
savefig(cost_fig, "dfl_policy_cost.png")

initial_policy = Policy("", "", DFLPolicy(initial_model, maximizer))
policy = Policy("", "", DFLPolicy(model, maximizer))

v, _ = evaluate_policy!(initial_policy, validation_environments, 10)
v
mean(v)
v2, _ = evaluate_policy!(policy, validation_environments, 10)
v2
mean(v2)

policies = generate_policies(b)
lazy = policies[1]
greedy = policies[2]
v3, _ = evaluate_policy!(lazy, validation_environments, 10)
mean(v3)
v4, _ = evaluate_policy!(greedy, validation_environments, 10)
mean(v4)

mean(
    map(validation_environments) do env
        v, y = generate_anticipative_solution(b, env; reset_env=true)
        return v
    end,
)

env = test_environments[4]
vv, data = evaluate_policy!(policy, env)
fig = DVSP.plot_epochs(data)
# savefig(fig, "dfl_policy_example.png")

vva, y = generate_anticipative_solution(b, env; reset_env=true)
DVSP.plot_epochs(y)

b2 = DynamicVehicleSchedulingBenchmark(; max_requests_per_epoch=20)
dataset2 = generate_dataset(b2, 10)
environments2 = generate_environments(b2, dataset2)

-mean(evaluate_policy!(policy, environments2)[1])
-mean(evaluate_policy!(greedy, environments2)[1])
-mean(evaluate_policy!(lazy, environments2)[1])
-(mean(map(e -> generate_anticipative_solution(b2, e; reset_env=true)[1], environments2)))

DVSP.plot_epochs(evaluate_policy!(policy, first(environments2))[2])

_, greedy_data = evaluate_policy!(greedy, first(environments2))
_, lazy_data = evaluate_policy!(lazy, first(environments2))
_, dfl_data = evaluate_policy!(policy, first(environments2))
_, anticipative_data = generate_anticipative_solution(
    b2, first(environments2); reset_env=true
)

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
