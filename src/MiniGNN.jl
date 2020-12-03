module MiniGNN

#  to import back in python https://stackoverflow.com/questions/33811864/how-do-i-within-python-retrieve-a-sparse-matrix-stored-by-julia-in-a-jld-file
#using GeometricFlux
using SparseArrays
using LinearAlgebra: issymmetric
using GraphSignals: adjacency_matrix

using Flux
using Functors: @functor, functor, fmap
using Flux: glorot_uniform, glorot_normal
using Flux: MaxPool, relu, Dropout

include("graph_layers.jl")
export make_rescaled_mat, adddims, process_data, create_batch

function make_rescaled_mat(adj_mat::AbstractMatrix{R}) where R<: Real
    deg = vec(sum(adj_mat,dims=1))
    denom = 1 ./ .√(1 .+ deg)
    denom .* permutedims(denom) .*adj_mat
end

function make_rescaled_mat(g::SparseMatrixCSC{R}) where R<:Real
    make_rescaled_mat(Matrix{Float32}(adjacency_matrix(g)))
end

function adddims(x::AbstractArray,k::Integer)
    @assert k<= ndims(x)+1
    add_last = reshape(x, (size(x)...,1))
    perm = Int[i for i = 1:ndims(add_last)]
    perm[k] = ndims(add_last)
    for i = k+1:ndims(add_last)
        perm[i] -= 1
    end
    permutedims(add_last,perm)
end

function process_data(data::AbstractArray)
    d = permutedims(data)
    return adddims(d,1)
end

function create_batch(bs::Int,train_loader)
    batch = []
    labels = []
    for (b,l) in train_loader
        batch = b[:,1:bs]
        labels = l[:,1:bs]
        break
    end
    batch, labels
end



end # module
