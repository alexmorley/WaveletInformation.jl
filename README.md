# WaveletInformation - WIP
[![Build](https://travis-ci.com/alexmorley/WaveletInformation.jl.svg?token=J1NxBfxGFhAmxxjYjuHi&branch=master)](https://travis-ci.com/alexmorley/WaveletInformation.jl)

No guarantee this is correct yet. 

Implemenents the WI method here: [Extracting information in spike time patterns with wavelets and information theory](http://jn.physiology.org/content/113/3/1015) Vítor Lopes-dos-Santos, Stefano Panzeri, Christoph Kayser, Mathew E Diamond, Rodrigo Quian Quiroga

And a "Wavelet PCA" method (weighting principal components based on their information.
Information can be measured using non-Gaussianity can be measured using K-S tests as described here: [Unsupervised Spike Detection and Sorting with Wavelets and Superparamagnetic Clustering](http://authors.library.caltech.edu/13699/1/QUInc04.pdf) or using GMM to fit multiple guassians and measuring the seperability of the peaks (which is so far the way I have implemented - it works better but is much more computationally expensive).

To Do:
- [ ] Double check against mlab for correctness
- [ ] Ensure type stability
- [ ] Add some docstrings
- [ ] Pure julia classifcation
