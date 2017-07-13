function mutual_info(response, class_id)
    nstim = length(unique(class_id));
    axis_MI= zeros(size(response,2));
    nbins_dist = 10;

    for axis_i in 1:size(response,2)
        if var(response[:,axis_i])>0
            histcounts = fit(Histogram,response[:,axis_i],nbins=nbins_dist,
        closed=:left).edges[1];
        else
            histcounts = fit(Histogram,0*response[:,axis_i],nbins=nbins_dist,
        closed=:left).edges[1];
        end

        probabilites = zeros(nstim,length(histcounts)-1);
        for stim_i = 1:nstim
            probabilites[stim_i,:] = fit(Histogram,response[class_id.==stim_i,axis_i],
        histcounts, closed=:left).weights;
        end

        probabilites = probabilites./sum(sum(probabilites));

        MI_aux = 0.;
        for bin_i = 1:length(histcounts)-1
            p_value = sum(probabilites[:,bin_i]); # response probability
            for stim_i = 1:nstim
                p_stim = sum(probabilites[stim_i,:]); # stimulus probability
                if probabilites[stim_i,bin_i]>0
                    PrGs = probabilites[stim_i,bin_i]; # response probability given stim
                    MI_aux = MI_aux + PrGs*log2(PrGs/(p_value*p_stim)); # MI
                end
            end
        end

        axis_MI[axis_i] = MI_aux;
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
