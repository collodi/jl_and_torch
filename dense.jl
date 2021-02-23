using Flux, CUDA
using Flux: Dense
using NNlib: relu, leakyrelu, sigmoid
using Statistics: mean
using Formatting: printfmtln
using DelimitedFiles

include("./data.jl")

data_dir = "./data/rand_tiny"

X = read_bods(data_dir)
Y = read_caps(data_dir)

tn_X = Flux.batch(X) |> gpu
tn_Y = Flux.batch(Y) |> gpu

println("data loaded")

tn_data = Flux.Data.DataLoader((tn_X, tn_Y), batchsize=1, shuffle=false)

m = Flux.Chain(Flux.flatten,
			   Dense(8, 4, relu),
			   Dense(4, 1, sigmoid)
			   ) |> gpu

# weights from a successful pytorch run
#w1 = [-0.0149  0.1073 -0.0076 -0.2996 -0.3232 -0.3424  0.0279 -0.1135;
#	   0.2696  0.1876  0.2073  0.1008  0.0418  0.0079 -0.1303  0.2748;
#	   0.0248  0.1720 -0.0438  0.2737 -0.3214  0.1305 -0.2084  0.1921;
#	  -0.0360 -0.0353 -0.0104 -0.2893  0.2577  0.0078 -0.1566 -0.0031]
#b1 = [-0.0634, 0.1410, 0.2333, 0.3425]
#
#w2 = [ 0.1557 -0.1028 -0.4271  0.3426]
#b2 = [-0.0866]

#m = Flux.Chain(Flux.flatten,
#			   Dense(8, 4, relu;
#					 initW = (a, b) -> w1,
#					 initb = a -> b1),
#			   Dense(4, 1, sigmoid;
#					 initW = (a, b) -> w2,
#					 initb = a -> b2)
#			   ) |> gpu

ps = Flux.params(m)
opt = Flux.Optimise.Descent(1e-2)
loss(x, y) = Flux.Losses.mse(m(x), y)

function train_err()
	y, ŷ = tn_Y, m(tn_X)
	return Flux.Losses.mae(ŷ, y)
end

println("epoch 0, mae $(train_err())")
for epoch = 1:500
	Flux.Optimise.train!(loss, ps, tn_data, opt)

	if epoch % 100 == 0
		println("epoch $epoch, mae $(train_err())")
	end
end
