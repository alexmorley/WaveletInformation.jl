# Install scikit-learn if not installed
ENV["PYTHON"]=""
using Pkg
Pkg.build("PyCall")

import PyCall
PyCall.pyimport_conda("sklearn", "scikit-learn")

using WaveletInformation
using Base.Test
