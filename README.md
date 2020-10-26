## Focal Loss with Label Smoothing
Label Smoothing applied in Focal Loss \
This library is based on the below papers.

- [Focal Loss for Dense Object Detection](https://arxiv.org/pdf/1708.02002.pdf).
- [When Does Label Smoothing Help?](https://arxiv.org/pdf/1906.02629.pdf)

# How to use
```
criteria = FocalLossWithSmoothing(num_classes)

logits = model(inputs)
loss = criteria(logits, labels)

optim.zero_grad()
loss.backward()
optim.step()
```
