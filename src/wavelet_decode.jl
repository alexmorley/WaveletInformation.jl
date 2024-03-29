export WIopts
export decode_spikecounts, decode_spiketimes

mutable struct WIopts
    nsurr::Int  # number of surrogates for computing shuffling distribution
    nscales::Union{Int, Symbol}
    percentile::Int # percentile of surrogate distribution for significance
    maxwvcoefs::Int  # maximum number of coefs to use
    minwvcoefs::Int  # minimum number of coefs to use
end

WIOpts() = WIopts(50, :max, 95, 25, 2)

function wavedec2d(clips::Array{T,2},
                   wavtype = WT.haar,
                   N = floor(Int,log2(size(clips,2)))
                  ) where T<:AbstractFloat
    # level of wavelet transform
    wt = wavelet(wavtype)
    # intialise array
    components_all = zeros(clips)
    for i in 1:size(components_all,1)
        components_all[i,:] = dwt(clips[i,:], wt, N)
    end
    return components_all::Array{T,2}
end

@sk_import naive_bayes: GaussianNB
function classify(sample, training, class_id_training,
                  classifier = GaussianNB())
    ScikitLearn.fit!(classifier, training[:,:], class_id_training[:,:])
    ScikitLearn.predict(classifier, sample)
end

function decode_spikecounts(actmatrix, class_id;
                            cross_val=LOOCV)
    spkcount = sum(actmatrix,dims=2)
    class_labels = unique(class_id)
    nclasses = length(class_labels)
    ntrials = length(spkcount)

    dec_output = Tuple{Array{Int64,1},Array{Int64,1}}[]

    ## Decode Spike Counts
    for trainingtrials in cross_val(ntrials)
        testtrials = setdiff(1:ntrials, trainingtrials)
        sample = spkcount[testtrials,1:1];
        training = spkcount[trainingtrials, 1:1];
        class_id_training = class_id[trainingtrials]
        clf = GaussianNB([1/nclasses for n in 1:nclasses])

        push!(dec_output, (class_id[testtrials], classify(sample, training, class_id_training, clf)))
    end

    return dec_output
end

function decode_spiketimes(actmatrix, class_id, opts::WIopts;
                           fast=false::Bool,
                           cross_val=LOOCV)
    class_labels = unique(class_id);
    nclasses = length(class_labels);
    ntrials = length(class_id); 
    opts.nscales = opts.nscales == :max ?
    floor(Int,log2(size(actmatrix,2))) : opts.nscales
    L = [1; [2^n for n in 1:opts.nscales]];

    dec_output = Tuple{Array{Int64,1},Array{Int64,1}}[]

    wvmatrix = wavedec2d(actmatrix, WT.haar, opts.nscales)

    fast && (MI_bias = mutual_info_thresh(wvmatrix, class_id,
                                          opts.nsurr, opts.percentile, L))
    for trainingtrials in cross_val(ntrials)
        testtrials = setdiff(1:ntrials, trainingtrials)
        wv_coefs = wvmatrix[trainingtrials,:]
        class_id_training = class_id[trainingtrials]
        fast || (MI_bias = mutual_info_thresh(wv_coefs, class_id_training,
                                              opts.nsurr, opts.percentile, L))
        selected_wvcoefs,_ = select_coefs(wv_coefs, class_id_training, MI_bias, opts.minwvcoefs,
                                          opts.maxwvcoefs)

        training = wv_coefs[:,selected_wvcoefs]
        sample   = wvmatrix[testtrials,selected_wvcoefs];
        clf      = GaussianNB([1/nclasses for n in 1:nclasses])

        push!(dec_output, (class_id[testtrials],
                           classify(sample,training,class_id_training, clf)
                          ))
    end

    return dec_output
end
