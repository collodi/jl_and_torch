# jl_and_torch

Different behaviors on PyTorch vs. Juia Flux

---

### Outputs

Flux
```
$ julia dense.jl
data loaded
epoch 0, mae 0.6247979
epoch 100, mae 0.062492933
epoch 200, mae 0.05447808
epoch 300, mae 0.053329896
epoch 400, mae 0.052486144
epoch 500, mae 0.05169717
```

PyTorch
```
$ python dense.py
epoch 0, mae 0.227466
epoch 100, mae 0.006111
epoch 200, mae 0.000271
epoch 300, mae 0.000006
epoch 400, mae 0.000000
```

---

### Is it weight init?

If you run the python code many times, you sometimes get errors that don't improve.
From my testing, this comes from bad initialization.
I've confirmed this by re-running the code with the same weight init values.

I've suspected that this might be the case for Flux.
However, copying a good weight init value from PyTorch to Flux does not improve errors.
If you'd like to test this, uncomment the custom weight model in `dense.jl`.
