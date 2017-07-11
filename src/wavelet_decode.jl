export WIopts
export decode_spikecounts, decode_spiketimes

mutable struct WIopts
    nsurr::Int  # number of surrogates for computing shuffling distribution
    nscales::Union{Int, Symbol}
	percentile::Int # percentile of surrogate distribution for significance
    maxwvcoefs::Int  # maximum number of coefs to use
    minwvcoefs::Int  # minimum number of coefs to use
end



function wavedec2d{T<:AbstractFloat}(clips::Array{T,2},
                   wavtype = WT.haar)
    # level of wavelet transform
    wt = wavelet(wavtype)
    N = floor(Int,log2(size(clips,2)))
    # intialise array
    components_all = zeros(clips)
    for i in 1:size(components_all,1)
        components_all[i,:] = dwt(clips[i,:], wt, N)
    end
    return components_all::Array{T,2}
end

@sk_import naive_bayes: GaussianNB
function classify(sample, training, class_id_training, classifier = GaussianNB())
    ScikitLearn.fit!(classifier, training[:,1:1], class_id_training[:,1:1])
    ScikitLearn.predict(classifier, sample)
end

function decode_spikecounts(actmatrix, class_id)
    spkcount = sum(actmatrix,2)
    class_labels = unique(class_id)
    nclasses = length(class_labels)
    ntrials = length(spkcount)
    
    dec_output = zeros(Int, ntrials)

    ## Decode Spike Counts
    for (trial_i, trainingtrials) in enumerate(LOOCV(ntrials))
        sample = spkcount[trial_i];
        training = spkcount[trainingtrials];
        class_id_training = class_id[trainingtrials]
        dec_output[trial_i] = classify(sample,training,class_id_training)[1]
    end

    return dec_output
end

function decode_spiketimes(actmatrix, class_id, opts::WIopts)
    class_labels = unique(class_id);
    nclasses = length(class_labels);
    ntrials = length(class_id); 
    opts.nscales = opts.nscales == :max ?
        floor(Int,log2(size(actmatrix,2))) : opts.nscales
    L = [1; [2^n for n in 1:opts.nscales]];
    
    dec_output = zeros(Int, ntrials)
    
    wvmatrix = wavedec2d(actmatrix)
    for (trial_i, trainingtrials) in enumerate(LOOCV(ntrials))
        wv_coefs = wvmatrix[trainingtrials,:]
        class_id_training = class_id[trainingtrials]
        MI_bias = mutual_info_thresh(wv_coefs, class_id_training, opts.nsurr, opts.percentile, L)
        selected_wvcoefs = select_coefs(wv_coefs, class_id_training, MI_bias, opts.minwvcoefs,
                                       opts.maxwvcoefs)

        training = wv_coefs[:,selected_wvcoefs]
        sample = wvmatrix[trial_i:trial_i,selected_wvcoefs];

        dec_output[trial_i] = classify(sample,training,class_id_training)[1]
    end
    
    return dec_output
end