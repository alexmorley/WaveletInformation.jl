function modulation_score(n::NeuronResponse, decoders::Array{ResponseDecoder,1},
                          num_spikes, num_trials, num_surrogates)
    results = Dict(d.name=>Float64[] for d in decoders)
    for _ in 1:num_surrogates
        spikes = [n.get_spikes(num_spikes) for _ in 1:num_trials]
        activity_matrix = reduce(hcat,fit(Histogram, s, τ).weights for s in spikes)
        # split into before and after "trials"
        mid = floor(Int,size(activity_matrix,1)/2)
        activity_matrix_split    = [activity_matrix[1:mid,:]';
                                    activity_matrix[mid+1:end,:]']
        before_after_identifiers = [zeros(size(activity_matrix,2));
                                    ones(size(activity_matrix,2))]
        for decoder in decoders
            performance = decoder.decode(activity_matrix_split, 
                                         before_after_identifiers)
            push!(results[decoder.name],performance)
        end
    end
    return results
end

function modulation_score(responses::Array{NeuronResponse, 1},
    decoders::Array{ResponseDecoder,1},
                          num_spikes, num_trials, num_surrogates)
    results = Dict(r.name => 
                   modulation_score(r, decoders,
                                    num_spikes, num_trials, num_surrogates)
                   for r in responses)
end
