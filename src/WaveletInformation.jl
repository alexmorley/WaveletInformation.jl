module WaveletInformation

using Reexport
@reexport using MLBase
using ScikitLearn
using Wavelets
using Combinatorics
using MultivariateStats
using GaussianMixtures
using Suppressor

include("wavelet_decode.jl")
include("MI.jl")
include("wpca.jl")

end
