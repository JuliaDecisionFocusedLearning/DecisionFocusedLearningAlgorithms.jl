using DecisionFocusedLearningAlgorithms
using DecisionFocusedLearningBenchmarks

using Flux
using MLUtils
using Plots

b = ArgmaxBenchmark()
initial_model = generate_statistical_model(b)
maximizer = generate_maximizer(b)
dataset = generate_dataset(b, 100)
train_dataset, val_dataset, test_dataset = splitobs(dataset; at=(0.3, 0.3, 0.4))

algorithm = PerturbedImitationAlgorithm(;
    nb_samples=20, ε=0.05, threaded=true, training_optimizer=Adam()
)

model = deepcopy(initial_model)
history = train!(algorithm, model, maximizer, train_dataset, val_dataset; epochs=50)
x, y = get(history, :training_loss)
plot(x, y; xlabel="Epoch", ylabel="Training Loss", title="Training Loss over Epochs")
