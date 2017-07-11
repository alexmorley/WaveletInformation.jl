module WaveletInformation

using Reexport
@reexport using MLBase
using ScikitLearn
using Wavelets

include("wavelet_decode.jl")
include("MI.jl")

end
