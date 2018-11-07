function raster_plot!(ax, spikes::Array{Array{T,1},1}) where T
    for (trial,spike_train) in enumerate(spikes)
        ax[:vlines](spike_train,trial,trial-1)
    end
end

function raster_hist(spikes)
    f,axs = subplots(2,1)
    raster_plot!(axs[1],spikes)
    axs[2][:hist](reduce(vcat,spikes),bins=50)
end


