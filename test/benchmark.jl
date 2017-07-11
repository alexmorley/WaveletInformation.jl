using WaveletInformation
using StatsBase

r = MersenneTwister(5)
testmat = rand(r, 100, 1024)
testids = sample(r, [1,2], 100)

@time WaveletInformation.mutual_info(testmat,testids)
@time WaveletInformation.mutual_info(testmat,testids)

@show  WaveletInformation.mutual_info(testmat,testids)[1:5]

L = [1; [2^n for n in 1:10]];
@time WaveletInformation.mutual_info_thresh(testmat,testids,5, 95, L)
