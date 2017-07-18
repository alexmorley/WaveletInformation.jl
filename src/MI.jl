function fasthist!(h, v, edg) 
    n = length(edg)-1
    for x in v
        i = searchsortedfirst(edg, x)-1
        if 1 <= i <= n
            h[i] .+= 1.
        end
    end
end

function mi(probabilites)
    a = 0.
    @inbounds @fastmath for bin_i = 1:size(probabilites,2)
        p_value = sum(probabilites[:,bin_i]); # response probability
        for stim_i = 1:size(probabilites,1)
            p_stim = sum(probabilites[stim_i,:]); # stimulus probability
            if probabilites[stim_i,bin_i]>0
                PrGs = probabilites[stim_i,bin_i]; # response probability given stim
                a +=  PrGs*log2(PrGs/(p_value*p_stim)); # MI
            end
        end
    end
    return a
end

function mutual_info(response, class_id)
    nstim = length(unique(class_id));
    nfet = size(response,2)
    axis_MI= zeros(Float64,nfet);
    nbins_dist = 10::Int
    histcounts = StatsBase.histrange(extrema(response)..., nbins_dist,
        :left)::StepRangeLen
    probabilites = zeros(Float64,nstim,length(histcounts)-1); 
    for axis_i in 1:nfet
        resp = view(response, :, axis_i)

        for stim_i = 1:nstim
            fasthist!(view(probabilites,stim_i,:),resp[class_id.==stim_i],
                histcounts);
        end
        probabilites ./= sum(probabilites);
        
        axis_MI[axis_i]=mi(probabilites)
        
        fill!(probabilites,0.0)
    end
    return axis_MI
end

function surrogate_MI_coefs(wvmatrix, class_id, nsurr)
    MI_coefs_surrogate = zeros(nsurr,size(wvmatrix,2))
    for surr_i = 1:nsurr
        class_id_surrogate = shuffle(class_id);
        MI_coefs_surrogate[surr_i,:] = mutual_info(wvmatrix,class_id_surrogate);
    end
    return MI_coefs_surrogate
end

function level_per_thresh(MI_coefs_surrogate,  percentile, L)
    MI_thresholds = zeros(size(MI_coefs_surrogate,2))

	levelbounds = cumsum([0; L[1:end-1]]);
    levelthres = zeros(1,length(L)-1);
    for level_i=1:length(L)-1
        level_coefs = levelbounds[level_i]+1:levelbounds[level_i+1];
        level_MI_nulldist = MI_coefs_surrogate[:,level_coefs];
        level_MI_nulldist = level_MI_nulldist[:];
        levelthres[level_i] = StatsBase.percentile(level_MI_nulldist,percentile);
        MI_thresholds[level_coefs] = levelthres[level_i];
    end
    return MI_thresholds
end

function mutual_info_thresh(wvmatrix, class_id, nsurr, percentile, L)
    MI_coefs_surrogate = surrogate_MI_coefs(wvmatrix, class_id, nsurr)
    MI_thresholds = level_per_thresh(MI_coefs_surrogate,  percentile, L)
	return MI_thresholds
end

# rename to mi_select
function select_coefs(wv_coefs, class_id_training, MI_bias, minwvcoefs, maxwvcoefs;
    visualise=false)
    MI = mutual_info(wv_coefs, class_id_training)
    
    unbiased_MI = MI .- MI_bias;
    selected_wvcoefs = find(unbiased_MI.>0)

    if length(selected_wvcoefs) > maxwvcoefs
        selected_wvcoefs = sortperm(unbiased_MI, rev=true)[1:maxwvcoefs]
    elseif length(selected_wvcoefs) < minwvcoefs
        selected_wvcoefs = sortperm(unbiased_MI, rev=true)[1:minwvcoefs]
    end
    
    if visualise
        nonselectedwvcs = setdiff(1:length(MI), selected_wvcoefs)
        scatter(nonselectedwvcs, MI[nonselectedwvcs], color="k", alpha=0.5)
        scatter(selected_wvcoefs, MI[selected_wvcoefs], color="r", alpha=0.5)
    end
    
    return selected_wvcoefs,(MI,unbiased_MI)
end
