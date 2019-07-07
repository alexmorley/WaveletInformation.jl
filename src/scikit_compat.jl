export WaveletClassifier

using ScikitLearnBase
using ScikitLearn.CrossValidation: cross_val_score, StratifiedKFold
import ScikitLearn.@declare_hyperparameters

DIR = @__DIR__

include("$DIR/../tutorial/decoders.jl")
include("$DIR/../tutorial/neurons.jl")
include("$DIR/../tutorial/tutorial_lib.jl")

mutable struct WaveletClassifier <: BaseClassifier
    nsurr::Int
    nscales::Union{Symbol, Int}
    percentile::Int
    maxcoefs::Int
    mincoefs::Int
    model
    chosen_features
    mutual_information
    unbiased_mutual_information
    WaveletClassifier(nsurr, nscales, percentile, maxcoefs, mincoefs) = new(nsurr, nscales, percentile, maxcoefs, mincoefs)
end

function WaveletClassifier(; nsurr=100, nscales=:max, percentile=95, maxcoefs=5, mincoefs=2)
    WaveletClassifier(nsurr, nscales, percentile, maxcoefs, mincoefs)
end

@declare_hyperparameters(WaveletClassifier, Symbol[:nsurr, :nscales, :percentile, :maxcoefs, :mincoefs])

import ScikitLearnBase.is_classifier
is_classifier(clf::WaveletClassifier) = true

function ScikitLearnBase.fit!(lr::WaveletClassifier, X::AbstractArray{XT},
                              y::AbstractArray{yT}) where {XT, yT}
    decomposed_matrix = decompose(X)
    opts = WIopts(lr.nsurr, lr.nscales, lr.percentile, lr.maxcoefs, lr.mincoefs)
    feature_select,(mi,umi)  = WaveletInformation.mi_select(decomposed_matrix,
                                                        y, opts)
    lr.chosen_features = feature_select
    lr.mutual_information = mi
    lr.unbiased_mutual_information = umi
    
    X = decomposed_matrix[:,feature_select]
    y = y
    n = length(unique(y))
    lr.model = GaussianNB(priors=[1/n for _ in 1:n])
    ScikitLearn.fit!(lr.model, X, y)
    return lr
end

function ScikitLearnBase.predict(lr::WaveletClassifier, X)
    X2 = decompose(X)[:, lr.chosen_features]
    ScikitLearn.predict(lr.model, X2)
end
