using TensorBoardLogger, Logging, Random

lg = TBLogger("tensorboard_logs/run"; min_level=Logging.Info)

struct sample_struct
    first_field
    other_field
end

with_logger(lg) do
    for i in 1:100
        x0 = 0.5 + i / 30
        s0 = 0.5 / (i / 20)
        edges = collect(-5:0.1:5)
        centers = collect(edges[1:(end - 1)] .+ 0.05)
        histvals = [exp(-((c - x0) / s0)^2) for c in centers]
        data_tuple = (edges, histvals)
        data_struct = sample_struct(i^2, i^1.5 - 0.3 * i)

        @info "test" i = i j = i^2 dd = rand(10) .+ 0.1 * i hh = data_tuple
        @info "test_2" i = i j = 2^i hh = data_tuple log_step_increment = 0
        @info "" my_weird_struct = data_struct log_step_increment = 0
        @debug "debug_msg" this_wont_show_up = i
    end
end

Dict(:loss => (s, i) -> s + i, :accuracy => (s, i) -> s - i)
