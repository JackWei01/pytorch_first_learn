import torch


#查看torch 下的里有哪些工具套件
dir(torch)

'''
['AVG',
 'AggregationType',
 'AliasDb',
 'Any',
 'BenchmarkConfig'
 'BoolStorage',
 'ByteStorage',
 'ByteTensor',
 'CallStack',
 'Callable'
 'CharTensor',
 'ClassType',
 'Code',
...
'''
#torch 下的 CUDA 里有哪些工具套件
dir(torch.cuda)
'''
'cudaStatus',
 'cudart',
 'current_blas_handle',
 'current_device',
 'current_stream',
 'default_generators',
 'default_stream',
 'device',
 'device_count',
 'device_of',
 'empty_cache',
 'get_allocator_backend',
...
'''
#此处的is_available 已经是最终的函数了
#help 查看 函数的说明文档
help(torch.cuda.is_available)

# b = range(1,10)
#
# print(dir(b))
#
# help(dir(b))