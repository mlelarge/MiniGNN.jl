using MiniGNN
using JLD, HDF5
# Beware! https://github.com/JuliaIO/JLD.jl/issues/234
using Flux.Data: DataLoader
using CUDA
using MiniFastai
using Flux: logitcrossentropy, onehotbatch, Descent, Dense, Dropout, relu, Chain
using SparseArrays
using MiniGNN: create_batch

@load "train_data.jld" train_data
#size Nsamples x V
@load "train_labels.jld" train_labels
@load "test_data.jld" test_data
@load "test_labels.jld" test_labels

#small = 1000
#train_data = train_data[1:small,:]
#train_labels = train_labels[1:small]
train_labels = onehotbatch(train_labels, 0:9)
#size Nlabels x Nsamples
test_labels = onehotbatch(test_labels,0:9)
d = load("List_graphs.jld")
L_g = d["L_g"]
sizes = d["sizes"]

bs = 100

train_loader = DataLoader((permutedims(train_data),train_labels), batchsize=bs, shuffle=true)
val_loader = DataLoader((permutedims(test_data),test_labels), batchsize=bs, shuffle=false)
databunch = Databunch(train_loader,val_loader)

batch, labels = create_batch(100,train_loader)
input = convert(Array{Float32,3},process_data(batch))

p=0.5
mat1 = MiniGNN.make_rescaled_mat(L_g[1])
layer1 = MiniGNN.GraphConv(mat1,1=>32,5)
mp = MiniGNN.GraphMaxPool(4)
mat2 = MiniGNN.make_rescaled_mat(L_g[3])
layer2 = MiniGNN.GraphConv(mat2,32=>64,5)
dim_fc = Int(64*size(mat2)[1]/4)
model = Chain(layer1 , mp, layer2, mp, MiniGNN.graphflatten, Dense(dim_fc,512, relu), Dropout(p), Dense(512,10)) 

learning_rate = 0.05
SGD = Descent(learning_rate)
loss(pred,y) = logitcrossentropy(pred,y)

learner_gnn = Learner(model,SGD,loss,databunch,cbs=(AvgStatsCallback(),))
transform = x -> convert(Array{Float32,3},process_data(x))
fit!(learner_gnn,30,transform)

model(input)
#one batch
using Flux
using Zygote: pullback, update!
ps = params(model)
            #source https://fluxml.ai/Flux.jl/stable/training/training/
            # https://fluxml.ai/Zygote.jl/latest/adjoints/#Pullbacks-1
y_pred, back_net = pullback(() -> model(input), ps)
current_loss, back_loss = pullback(y -> loss(y,labels),y_pred)
            #current_loss, back = pullback(() -> los_f(mod(xb),yb), ps)
gs = back_net(back_loss(1)[1])#back(one(current_loss))
update!(SGD,ps,gs)
