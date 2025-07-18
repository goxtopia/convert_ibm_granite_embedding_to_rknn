# convert_ibm_granite_embedding_to_rknn

## why do this
On a chip like RK3588 that has an NPU, running some emb model(feature extraction) on the device is such a great deal.
It's faster, costs less power, and sounds pretty cool too!
Also, this is a little demo showing you have to split a slightly complex model, thus we could make the most computationally costly part run on NPU!
