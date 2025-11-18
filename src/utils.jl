function get_info(sample)
    return (; instance=sample.info)
end

function get_state(sample)
    return (; instance=sample.info.state)
end
