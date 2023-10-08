import torch




outputs =torch.tensor([
    [0.1,0.2],
    [0.3,0.4]
])

print(outputs.argmax(1)) #对outputs 横向看  【0.1,0.2】一组比较  [0.3,0.4]一组比较
print(outputs.argmax(0)) #对outputs 纵向看  【0.1,0.3】一组比较  [0.2,0.4]一组比较

preds = outputs.argmax(1)
targets = torch.tensor([0,1])#注意类型都是 tensor

print((preds == targets).sum())

#计算
# if (preds==targets):

#计算查准率


