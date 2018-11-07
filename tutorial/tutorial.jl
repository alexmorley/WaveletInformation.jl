## Plotting Functions
using PyPlot
using WaveletInformation
using Distributions, StatsBase

if !isinteractive() 
    DIR = @__DIR__ 
else
    # assume root project
    DIR = "./tutorial"
end
include("$DIR/decoders.jl")
include("$DIR/neurons.jl")
include("$DIR/plotting.jl")

resolution = 0.001 # 1 ms
τ = range(-8.192,
          stop=8.192,
          step=resolution)
n_trials = 200

neuron_responses = [
                    NeuronResponse("Gaussian around stimulus",
                                   get_spikes_from_distribution(Normal(0,2))),
                    NeuronResponse("Gaussian offset from stimulus",
                                   get_spikes_from_distribution(Normal(2,2))),
                    NeuronResponse("Ramp after stimulus",
                                   N -> wsample(τ,ramp(τ),N)),
                    NeuronResponse("Oscillate after stimulus",
                                   N -> wsample(τ,oscillate(τ),N)),
                    NeuronResponse("Short response and refractory",
                                   N -> wsample(τ,refractory(τ),N))
                   ]

decoders = [
            ResponseDecoder("Spike Count",
                            spike_count_score),
            ResponseDecoder("PCA",
                            pca_score),
            ResponseDecoder("Wavelet Information",
                            (x,y)->wavelet_decoder_score(x,y,WIopts(50, :max,
                                                                    95, 5, 2)
                                                        ))
           ]


function raster_hist(n::NeuronResponse,num_spikes,num_trials)
    spikes = [n.get_spikes(num_spikes) for _ in 1:num_trials]
    raster_hist(spikes)
end

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

modulation_scores = modulation_score(neuron_responses[1:1],
                                     decoders[1:1],
                                     1000, #spikes
                                     100, #trials 
                                     10, #surrogates
                                    )

@time begin
num_spikes_array = [10 .^ (range(1,step=1,stop=5)) 5*
                    (10 .^ (range(1,step=1,stop=5)))]'[:]
modulation_scores_by_nspikes = [modulation_score(neuron_responses,
                                     decoders[1:1],
                                     num_spikes, #spikes
                                     100, #trials 
                                     10, #surrogates
                                    ) for num_spikes in num_spikes_array]
end
@time begin
num_trials_array = [10 .^ (range(1,step=1,stop=2)) 5*
                    (10 .^ (range(1,step=1,stop=2)))]'[1:end-1]
modulation_scores_by_num_trials = [modulation_score(neuron_responses[1:1],
                                     decoders[1:1],
                                     1000, #spikes
                                     num_trials, #trials 
                                     10, #surrogates
                                    ) for num_trials in num_trials_array]
end

decoder_colors = [[156,0,255]./256,[9,178,85]./256,[255,173,25]./256]
f,axs = subplots(3,length(neuron_responses),
                 figsize = (10,5),
                 gridspec_kw = Dict(:height_ratios=>[1,1,2])
                )
for (ni,neuron_response) in enumerate(neuron_responses)
    raster_ax = axs[1,ni]
    spikes = [neuron_response.get_spikes(100) for _ in 1:10]
    raster_plot!(raster_ax, spikes)
    #
    histogram_ax = axs[2,ni]
    histogram_ax[:hist](reduce(vcat,spikes),bins=50)
    #
    modulation_ax = axs[3,ni]
    modulation = [m[neuron_response.name]
                  for m in modulation_scores_by_nspikes]
    for (di,decoder) in enumerate(decoders)
        m = [m[decoder.name] for m in modulation]
        mean_m = [mean(mx) for mx in m]
        modulation_ax[:plot](num_spikes_array, mean_m,
                             label=decoder.name,
                             color=decoder_colors[di])
        for (oi,obs) in enumerate(m)
            modulation_ax[:scatter](repeat(num_spikes_array[oi:oi], length(obs)),
                         obs,
                         s=0.8,
                         color=decoder_colors[di],
                         alpha=0.5)
        end
        modulation_ax[:set_xscale]("log")
    end
    modulation_ax[:set_ylim](0,1.1)
    if ni == 1
        modulation_ax[:set_xlabel]("Number of Spikes")
        modulation_ax[:set_ylabel]("Decoder Accuracy")
    end
end


function modulation_score(responses::Array{NeuronResponse, 1},
    decoders::Array{ResponseDecoder,1},
                          num_spikes, num_trials, num_surrogates)
    results = Dict(r.name => 
                   modulation_score(r, decoders,
                                    num_spikes, num_trials, num_surrogates)
                   for r in responses)
end

#=
    #
    x1 = spike_count_score(activity_matrix_split, before_after_identifiers)
    #
    opts = WaveletInformation.WIopts(50,:max, 95, 5, 2)
    x2 = wavelet_decoder_score(activity_matrix_split, before_after_identifiers,
                               WaveletInformation.WIopts(50,:max, 95, 5, 2))
    println("""
            Spike Count Score: $x1
            Wavelet Decoder Score: $x2
            """
           )
=#
