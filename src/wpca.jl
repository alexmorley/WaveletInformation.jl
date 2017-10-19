function rungmm{T}(component::Array{T,1}, ngauss::Int, covtype=:full, nIter=30, perr=false)    
    if perr
        mm = GMM(ngauss, Float64.(component[:,1:1]),
            kind=covtype, nIter=nIter)
    else
    @capture_err begin; @capture_out begin
        mm = GMM(ngauss, Float64.(component[:,1:1]),
            kind=covtype, nIter=nIter)
    end; end
    end
    return mm
end
    
function rungmm{T<:AbstractFloat}(norm_components::Array{T,2},
    ngauss::Int, covtype=:full, nIter=30, v=false, vv=false)
    ncomps = size(norm_components,2)
    mms = []
    for ncomp in 1:ncomps
        v && println("Fitting to $ncomp/$ncomps components")
        push!(mms, rungmm(norm_components[:,ncomp],
            ngauss, covtype, nIter, vv))
    end
    mms
end
    
function getsep(mm, ngauss::Int)
    sep = 0
    for (i,j) in collect(combinations(1:ngauss,2))
        sep += (abs(mm.μ[i] - mm.μ[j]) * sqrt(mm.w[i] * mm.w[j])) /
            sqrt((mm.Σ[i][1]*mm.Σ[i][1]) * (mm.Σ[j][1]*mm.Σ[j][1]))
    end
    return sep
end

function check_seps(seps::Vector{<:Real},verbose::Bool=false)
    sepnans = isnan.(seps)
    if any(sepnans)
        verbose & warn("Some NaNs in seperability scores, setting to zero")
        seps[sepnans] = zero(eltype(seps))
    end
    seps
end

function weightedpca(clips)
    #### options ####
    ngauss = 4 # number of gaussians to fit to the wavelet components
    v = true # verbosity
    vv = false # double verbosity
    sep_flag = true # calculate seperability?
    nIterGMM = 500 # niterations for GMM

    v && println("Starting weighted PCA...") 
    # apply harr wavelet to each waveform
    println("Applying Wavelets ...")
    wavelet_components = wavedec2d(clips, WT.haar)
    nof_components = size(wavelet_components,2)
    norm_components = zscore(wavelet_components,1) # (data-mean)/std

    if sep_flag
        println("Running GMM ...")
        # fit ngauss gaussians to each component
        mms = rungmm(norm_components, ngauss, :diag, nIterGMM, v, vv)
        # get seperability metric (i.e. how far from unimodal distribution)
        seps = getsep.(mms, [ngauss])
        seps = check_seps(seps,vv)
    else
        seps=ones(nof_components)
    end
    
    wpca = fit(PCA,norm_components.*seps');

    return wpca.proj
end
