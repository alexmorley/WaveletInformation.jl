using PyPlot

if !isinteractive() 
    DIR = @__DIR__ 
else
    # assume root project
    DIR = "./tutorial"
end

@everywhere begin
@everywhere @eval DIR=$DIR
using WaveletInformation
using Distributions, StatsBase, MultivariateStats, Wavelets

resolution = 0.001 # 1 ms
τ = range(-8.192, # TODO this shouldn't be global
          stop=8.192,
          step=resolution)
n_trials = 200

include("$DIR/decoders.jl")
include("$DIR/neurons.jl")
include("$DIR/tutorial_lib.jl")

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

num_spikes_array = [10 .^ (range(1,step=1,stop=5)) 5*
                    (10 .^ (range(1,step=1,stop=5)))]'[:]

num_trials_array = [10 .^ (range(1,step=1,stop=3)) 5*
                    (10 .^ (range(1,step=1,stop=3)))]'[1:end-1]
end

modulation_scores = modulation_score(neuron_responses[1:1],
                                     decoders,
                                     1000, #spikes
                                     100, #trials 
                                     10, #surrogates
                                    )

@time begin
modulation_scores_by_num_spikes = pmap(num_spikes->
                                       modulation_score(neuron_responses,
                                     decoders,
                                     num_spikes, #spikes
                                     100, #trials 
                                     10, #surrogates
                                    ), num_spikes_array)
end
#
@time begin
modulation_scores_by_num_trials = pmap(num_trials->
                                       modulation_score(neuron_responses,
                                     decoders,
                                     1000, #spikes
                                     num_trials, #trials 
                                     10, #surrogates
                                    ), num_trials_array)
end
#"/tmp/julia6n8pMj"


function plot_modulation!(ax, x, modulation, decoders)
    for (di,decoder) in enumerate(decoders)
        m = [m[decoder.name] for m in modulation]
        mean_m = [mean(mx) for mx in m]
        ax[:plot](x, mean_m,
                             label=decoder.name,
                             color=decoder_colors[di])
        for (oi,obs) in enumerate(m)
            ax[:scatter](repeat(x[oi:oi], length(obs)),
                         obs,
                         s=0.8,
                         color=decoder_colors[di],
                         alpha=0.5)
        end
        ax[:set_xscale]("log")
    end
    ax[:set_ylim](0,1.1)
end

include("$DIR/plotting.jl")
plt[:style][:use]("/home/data/.config/matplotlib/stylelib/alex.mplstyle")
ioff()
decoder_colors = [[156,0,255]./256,[9,178,85]./256,[255,173,25]./256]
f,axs = subplots(4,length(neuron_responses),
                 figsize = (7,5),
                 gridspec_kw = Dict(:height_ratios=>[2,2,3,3]),
                 sharey="row"
                )
for (ni,neuron_response) in enumerate(neuron_responses)
    raster_ax = axs[1,ni]
    spikes = [neuron_response.get_spikes(100) for _ in 1:10]
    raster_plot!(raster_ax, spikes)
    #
    histogram_ax = axs[2,ni]
    histogram_ax[:hist](reduce(vcat,spikes),bins=50)
    #
    modulation_spikes_ax = axs[3,ni]
    modulation = [m[neuron_response.name]
                  for m in modulation_scores_by_num_spikes]
    plot_modulation!(modulation_spikes_ax, num_spikes_array,
                     modulation, decoders)
    modulation_trials_ax = axs[4,ni]
    modulation = [m[neuron_response.name]
                  for m in modulation_scores_by_num_trials]
    plot_modulation!(modulation_trials_ax, num_trials_array,
                     modulation, decoders)
    #
    if ni == 1
        raster_ax[:set_xlabel]("Stimulus Onset (s)")
        raster_ax[:set_yticks]([])
        histogram_ax[:set_xlabel]("Stimulus Onset (s)")
        histogram_ax[:set_ylabel]("Spike Count")
        modulation_spikes_ax[:set_xlabel]("Number of Spikes")
        modulation_spikes_ax[:set_ylabel]("Decoder\nAccuracy")
        modulation_trials_ax[:set_xlabel]("Number of Trials")
        modulation_trials_ax[:set_ylabel]("Decoder\nAccuracy")
    else
        nothing
    end
end
f[:tight_layout]()
f[:subplots_adjust](hspace=0.5)
if true#false
f[:savefig]("/mnfs/vtad1/data/amorley_figs/Thesis/wavelet_decoder_comparison.svg",
           )
f[:savefig]("/mnfs/vtad1/data/amorley_figs/Thesis/wavelet_decoder_comparison.png",
           )
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
