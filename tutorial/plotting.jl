function raster_plot!(ax, spikes::Array{Array{T,1},1}) where T
    for (trial,spike_train) in enumerate(spikes)
        ax[:vlines](spike_train,trial,trial-1,
                   lw=0.3)
    end
end

function raster_hist(spikes)
    f,axs = subplots(2,1)
    raster_plot!(axs[1],spikes)
    axs[2][:hist](reduce(vcat,spikes),bins=50)
end

function raster_hist(n::NeuronResponse,num_spikes,num_trials)
    spikes = [n.get_spikes(num_spikes) for _ in 1:num_trials]
    raster_hist(spikes)
end
