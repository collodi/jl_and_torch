function read_bods(dir)
	fns = [d for d in readdir(dir, join=true) if endswith(d, ".bod")]
	return [read_bod(fn) for fn in fns]
end

function read_caps(dir)
	fns = [d for d in readdir(dir, join=true) if endswith(d, ".cap")]
	return [read_cap(fn) for fn in fns]
end

function read_bod(fn)
	X = zeros(4, countlines(fn))
	for (i, ln) in enumerate(eachline(fn))
		X[:, i] = parse.(Float32, split(ln)[2:end])
	end

	return X
end

function read_cap(fn)
	ln = readlines(fn)[2]
	return parse(Float32, split(ln)[1])
end
