# To be used to visualize loss across iterations

using DecisionFocusedLearningAlgorithms
using DecisionFocusedLearningBenchmarks

benchmark = ContextualStochasticArgmaxBenchmark()

anticipative_solver = generate_anticipative_solver(benchmark)
algorithm = DecisionFocusedLearningAlgorithms.MirrorDescent()

κ = 0.1
train_dataset_size = 5
nb_epochs          = 2
nb_iterations      = 2
seed               = 3

histories_r, _ = DecisionFocusedLearningAlgorithms.train_policy(
    algorithm, benchmark;
    dataset_size = train_dataset_size,
    epochs       = nb_epochs,
    iterations   = nb_iterations,
    seed         = seed,
    κ            = κ,
)


