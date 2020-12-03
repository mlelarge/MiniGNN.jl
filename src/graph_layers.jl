

struct GraphConv{A<:AbstractArray{Float32},B<:AbstractArray{Float32},C<:AbstractArray{Float32}}
    weight::A # Fout x Fin x k
    bias::B # Fout
    L::C #Graph operator V x V
    k::Integer
    #in_channel::Integer
    #out_channel::Integer
end

function GraphConv(L::AbstractArray,ch::Pair{<:Integer,<:Integer}, k::Integer; 
    init = glorot_normal, bias::Bool=false)
    b = bias ? init(ch[2]) : zeros(Float32,ch[2])
    GraphConv(init(ch[2], ch[1], k), b, L, k)
end

# see https://fluxml.ai/Flux.jl/stable/models/advanced/
# https://discourse.julialang.org/t/implementing-a-custom-layer/38632
@functor GraphConv #(weight, bias,)
Flux.trainable(c::GraphConv) = (c.weight,c.bias,)

function (c::GraphConv)(X::AbstractArray)
    fin, b, v = size(X) #Features_in, Batch, Vertices
    fout = size(c.weight,1)
    Y = view(c.weight,:,:,1) * reshape(X, fin, b*v) #Fout x B*V
    Z_prev = reshape(X, fin*b, v) #B*Fin x V
    Z = Z_prev * c.L #B*Fin x V
    Y += view(c.weight,:,:,2) * reshape(Z, fin, b*v) 
    for k = 3:c.k
        Z, Z_prev = 2*Z*c.L - Z_prev, Z #B*Fin x V
        Y += view(c.weight,:,:,k) * reshape(Z, fin, b*v) 
    end
    out = reshape(Y,fout, b, v) .+ c.bias
    relu.(out)
end

struct GraphMaxPool
    k::Integer
end

function (m::GraphMaxPool)(X::AbstractArray{T}) where {T<:Float32}
    #size(X) Feat x B x V
    permutedims(maxpool(permutedims(X,[3,1,2]),(m.k,)) , [2,3,1])#Feat x B x V/k 
end

function graphflatten(x::AbstractArray)
    #size(x) Feat x B x V
    x = permutedims(x, [1,3,2])# Feat x V x B
    return reshape(x, :, size(x)[end])# Feat*V x B
end