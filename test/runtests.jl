# Install scikit-learn if not installed
import PyCall
PyCall.pyimport_conda("sklearn", "scikit-learn")

using WaveletInformation
using Base.Test
