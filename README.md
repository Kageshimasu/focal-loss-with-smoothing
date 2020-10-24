## Focal Loss with Label Smoothing
Label Smoothing applied in Focal Loss

# How to use
```
criteria = FocalLossWithSmoothing(num_classes)

logits = model(inputs)
loss = criteria(logits, labels)

optim.zero_grad()
loss.backward()
optim.step()
```
