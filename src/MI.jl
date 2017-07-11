function fasthist!(h, v, edg) 
    n = length(edg)-1
    y = 1./length(v)
    for x in v
        i = searchsortedfirst(edg, x)-1
        if 1 <= i <= n
            h[i] += y
        end
    end
end

function mutual_info(response::Array{Float64,2}, class_id::Array{Int,1})
    nstim = length(unique(class_id))
    axis_MI= zeros(Float64, size(response,2))
    nbins_dist = 10 ::Int
    histcounts = StatsBase.histrange(extrema(response)..., nbins_dist, :left)
    probabilites = zeros(nstim,length(histcounts)-1); 
    
    for axis_i in 1:size(response,2)
        resp = view(response, :, axis_i)

        for stim_i = 1:nstim
            a = class_id.==stim_i 
            fasthist!(probabilites[stim_i,:], view(resp,a), histcounts)
        end

        probabilites ./= sum(probabilites);

        MI_aux = 0.;
        for bin_i = 1:length(histcounts)-1
            p_value = sum(probabilites[:,bin_i]); # response probability
            for stim_i = 1:nstim
                p_stim = sum(probabilites[stim_i,:]); # stimulus probability
                if probabilites[stim_i,bin_i]>0
                    PrGs = probabilites[stim_i,bin_i]; # response probability given stim
                    MI_aux += PrGs*log2(PrGs/(p_value*p_stim)); # MI
                end
            end
        end
        fill!(probabilites, 0.0)
        axis_MI[axis_i] = MI_aux;
    end
    return axis_MI
end

function mutual_info_thresh(wvmatrix, class_id, nsurr, percentile, L)
    MI_coefs_surrogate = zeros(nsurr,size(wvmatrix,2));
    for surr_i = 1:nsurr  
        class_id_surrogate = shuffle(class_id);
        MI_coefs_surrogate[surr_i,:] = mutual_info(wvmatrix,class_id_surrogate);
    end

    MI_thresholds = zeros(size(wvmatrix,2));
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
    
    return selected_wvcoefs
end
