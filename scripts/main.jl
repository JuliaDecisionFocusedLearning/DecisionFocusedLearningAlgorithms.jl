using DecisionFocusedLearningAlgorithms
using DecisionFocusedLearningBenchmarks
using MLUtils
using Statistics

struct KleopatraPolicy{M}
    model::M
end

function (m::KleopatraPolicy)(env)
    x, instance = observe(env)
    θ = m.model(x)
    return maximizer(θ; instance)
end

fyl_train_model(ArgmaxBenchmark(); epochs=1000)
baty_train_model(DynamicVehicleSchedulingBenchmark(; two_dimensional_features=false))
DAgger_train_model(DynamicVehicleSchedulingBenchmark(; two_dimensional_features=false))

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

metrics_callbacks = (;
    obj=(model, maximizer, epoch) ->
        mean(evaluate_policy!(fyl_policy, test_environments, 1)[1])
)

fyl_loss = fyl_train_model!(
    fyl_model, maximizer, train_dataset, val_dataset; epochs=100, metrics_callbacks
)

dagger_model = deepcopy(model)
dagger_policy = Policy("dagger", "", KleopatraPolicy(dagger_model))
metrics_callbacks = (;
    obj=(model, maximizer, epoch) ->
        mean(evaluate_policy!(dagger_policy, test_environments, 1)[1])
)
dagger_loss = DAgger_train_model!(
    dagger_model,
    maximizer,
    train_environments,
    validation_environments,
    anticipative_policy;
    iterations=10,
    fyl_epochs=10,
    metrics_callbacks,
)

plot(
    0:100,
    [fyl_loss.obj[1:end], dagger_loss.obj[1:end]];
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
