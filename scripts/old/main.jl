using DecisionFocusedLearningAlgorithms
using DecisionFocusedLearningBenchmarks
using MLUtils
using Statistics
using Plots

# ! metric(prediction, data_sample)

b = ArgmaxBenchmark()
initial_model = generate_statistical_model(b)
maximizer = generate_maximizer(b)
dataset = generate_dataset(b, 100)
train_dataset, val_dataset, _ = splitobs(dataset; at=(0.3, 0.3, 0.4))
res, model = fyl_train_model(
    initial_model, maximizer, train_dataset, val_dataset; epochs=100
)

res = fyl_train_model(StochasticVehicleSchedulingBenchmark(); epochs=100)
plot(res.validation_loss; label="Validation Loss")
plot!(res.training_loss; label="Training Loss")

kleopatra_train_model(DynamicVehicleSchedulingBenchmark(; two_dimensional_features=false))
DAgger_train_model(DynamicVehicleSchedulingBenchmark(; two_dimensional_features=false))

struct KleopatraPolicy{M}
    model::M
end

function (m::KleopatraPolicy)(env)
    x, instance = observe(env)
    θ = m.model(x)
    return maximizer(θ; instance)
end

b = DynamicVehicleSchedulingBenchmark(; two_dimensional_features=false)
dataset = generate_dataset(b, 100)
train_instances, validation_instances, test_instances = splitobs(
    dataset; at=(0.3, 0.3, 0.4)
)
train_environments = generate_environments(b, train_instances; seed=0)
validation_environments = generate_environments(b, validation_instances)
test_environments = generate_environments(b, test_instances)

train_dataset = vcat(map(train_environments) do env
    v, y = generate_anticipative_solution(b, env; reset_env=true)
    return y
end...)

val_dataset = vcat(map(validation_environments) do env
    v, y = generate_anticipative_solution(b, env; reset_env=true)
    return y
end...)

model = generate_statistical_model(b; seed=0)
maximizer = generate_maximizer(b)
anticipative_policy = (env; reset_env) -> generate_anticipative_solution(b, env; reset_env)

fyl_model = deepcopy(model)
fyl_policy = Policy("fyl", "", KleopatraPolicy(fyl_model))

callbacks = [
    Metric(:obj, (data, ctx) -> mean(evaluate_policy!(fyl_policy, test_environments, 1)[1]))
]

fyl_history = fyl_train_model!(
    fyl_model, maximizer, train_dataset, val_dataset; epochs=100, callbacks
)

dagger_model = deepcopy(model)
dagger_policy = Policy("dagger", "", KleopatraPolicy(dagger_model))

callbacks = [
    Metric(
        :obj, (data, ctx) -> mean(evaluate_policy!(dagger_policy, test_environments, 1)[1])
    ),
]

dagger_history = DAgger_train_model!(
    dagger_model,
    maximizer,
    train_environments,
    anticipative_policy;
    iterations=10,
    fyl_epochs=10,
    callbacks=callbacks,
)

# Extract metric values for plotting
fyl_epochs, fyl_obj_values = get(fyl_history, :val_obj)
dagger_epochs, dagger_obj_values = get(dagger_history, :val_obj)

plot(
    [fyl_epochs, dagger_epochs],
    [fyl_obj_values, dagger_obj_values];
    labels=["FYL" "DAgger"],
    xlabel="Epoch",
    ylabel="Test Average Reward (1 scenario)",
)

using Statistics
v_fyl, _ = evaluate_policy!(fyl_policy, test_environments, 100)
v_dagger, _ = evaluate_policy!(dagger_policy, test_environments, 100)
mean(v_fyl)
mean(v_dagger)

anticipative_policy(test_environments[1]; reset_env=true)
