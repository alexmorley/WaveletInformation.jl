mutable struct NeuronResponse
    name::String
    get_spikes::Function
end

function get_spikes_from_distribution(d)
    N -> (filter(x->first(τ)<x<last(τ), sort(rand(d,2N)[1:N])))
end

# smooth ramp
function ramp(τ)
    mt = maximum(τ)
    f(t) = begin
        if t < 0
            return 0.1
        else
            return t
        end
    end
    [f(t/mt) for t in τ]
end

# oscillate after
function oscillate(τ)
    mt = maximum(τ)
    f(t) = begin
        if t < 0
            return 0.5
        else
            return sin(2π*t)+0.5
        end
    end
    [f(t) for t in τ]
end

# short response and refractory period
function refractory(τ)
    mt = maximum(τ)
    f(t) = begin
        if t < 0
            return 0.5
        elseif t < 0.5
            return 0.9
        elseif t < 1.0
            return 0.1
        else
            return 0.5
        end
    end
    [f(t) for t in τ]
end

