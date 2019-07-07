mutable struct ResponseDecoder
    name::String
    decode::Function
    τ::AbstractArray
end

using ScikitLearn
@sk_import naive_bayes: GaussianNB
using ScikitLearn.CrossValidation: cross_val_score, StratifiedKFold
#@sk_import model_selection: StratifiedKFold
#import StatsBase.score
#function score(model_type, X, y)
#    model = model_type()
#    ScikitLearn.fit!(model, X, y)
#    accuracy = sum(ScikitLearn.predict(model, X) .== y) / length(y)
#    return accuracy
#end

function evaluate(model, X, y)
    return mean(cross_val_score(model, X, y, cv=CrossValidation.StratifiedKFold(y, shuffle=false, random_state=0, n_folds=5)))
end

function pca_score(activity_matrix_split, before_after_identifiers)
    pca = MultivariateStats.fit(PCA, Float64.(activity_matrix_split'))
    X = activity_matrix_split * pca.proj
    y = before_after_identifiers 
    return evaluate(GaussianNB(priors=[0.5,0.5]), X, y)
end

function spike_count_score(activity_matrix_split, before_after_identifiers)
    X = sum(activity_matrix_split,dims=2)
    y = before_after_identifiers 

    return evaluate(GaussianNB(priors=[0.5,0.5]), X, y)
end

function wavelet_decoder_score2(activity_matrix_split, before_after_identifiers; opts=WIOpts())
    X = activity_matrix_split
    y = before_after_identifiers
    clf = WaveletClassifier(nsurr=opts.nsurr, nscales=opts.nscales, 
                            maxcoefs=opts.maxwvcoefs, mincoefs=opts.minwvcoefs,
                            percentile=opts.percentile)
    return evaluate(clf, X, y)
end

function wavelet_decoder_score(activity_matrix_split, before_after_identifiers,
                               opts=WIopts())
    decomposed_matrix = decompose(activity_matrix_split)
    feature_select,(mi,umi)  = WaveletInformation.mi_select(decomposed_matrix,
                                                            before_after_identifiers,opts)
    wavelet_parameters = Dict(
                              "chosen_features" => feature_select,
                              "mutual_information" => mi,
                              "unbiased_mutual_information" => umi
                             )
    X = decomposed_matrix[:,feature_select]
    y = before_after_identifiers 
    
    return evaluate(GaussianNB(priors=[0.5,0.5]), X, y)
end

max_level(X) = floor(Int,log2(size(X,2)))

function decompose(X::Array{T,2}, args...; kwargs...) where T<:Integer
    decompose(Float64.(X), args...; kwargs...)
end

function decompose(X::Array{T,2}, W = wavelet(WT.haar);
                   level = max_level(X)
                  ) where T<:AbstractFloat
    # intialise array
    wX = zeros(size(X)...)
    for i in 1:size(wX,1)
        wX[i,:] = dwt(X[i,:], W, level)
    end
    return wX::Array{T,2}
end


