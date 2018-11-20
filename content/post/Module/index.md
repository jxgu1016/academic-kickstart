+++
title = "Module"
date = 2018-11-20T21:21:59+08:00
draft = false

# Authors. Comma separated list, e.g. `["Bob Smith", "David Jones"]`.
authors = ["Jiaxin Gu"]

# Tags and categories
# For example, use `tags = []` for no tags, or the form `tags = ["A Tag", "Another Tag"]` for one or more tags.
tags = ["PyTorch"]
categories = []

# Projects (optional).
#   Associate this post with one or more of your projects.
#   Simply enter your project's folder or file name without extension.
#   E.g. `projects = ["deep-learning"]` references 
#   `content/project/deep-learning/index.md`.
#   Otherwise, set `projects = []`.
# projects = ["internal-project"]

# Featured image
# To use, add an image named `featured.jpg/png` to your page's folder. 
[image]
  # Caption (optional)
  caption = ""

  # Focal point (optional)
  # Options: Smart, Center, TopLeft, Top, TopRight, Left, Right, BottomLeft, Bottom, BottomRight
  focal_point = ""
+++

# nn.Module
nn.Module是Pytorch的核心部分之一，它将单一的layer(Operation)组装成更抽象的网络(network)，类似caffe里面的定义网络结构的protxt文件。在构建我们自己的module时，我们都要继承nn.Module这个基类，常见的例子为：
```python
    import torch.nn as nn
    import torch.nn.functional as F

    class Model(nn.Module):
        def __init__(self):
            super(Model, self).__init__()
            self.conv1 = nn.Conv2d(1, 20, 5)
            self.conv2 = nn.Conv2d(20, 20, 5)

        def forward(self, x):
            x = F.relu(self.conv1(x))
            return F.relu(self.conv2(x))
```       
## 初始化\_\_init\_\_
```python
    def __init__(self):
        self._backend = thnn_backend
        self._parameters = OrderedDict()
        self._buffers = OrderedDict()
        self._backward_hooks = OrderedDict()
        self._forward_hooks = OrderedDict()
        self._forward_pre_hooks = OrderedDict()
        self._modules = OrderedDict()
        self.training = True
```
由于nn.Module继承了object类，所以实例化后的8个属性(attribute)都保存在\__dict\__中。这次我们重点研究是的其中的3个：\_buffers, \_parameters与\_modules，剩下的3个hooks在另外一篇笔记中分析。

首先关注的是OrderedDict，它是dict的子类，记住了内容添加的顺序，访问方法也与普通的dict一致。以\_parameters为例，OrderedDict的数据结构为：
```python
    OrderedDict([('conv1', Conv2d(...)), ('conv2', Conv2d(...))])
```
其中的value须为Module的子类(Conv2d等)。

## 赋值\_\_setattr\_\_
如前文所说，原生的Module类有且只有8个属性，那之后我们添加如Conv2d的子module是如何存放的呢？这就要看到重写后\_\_setattr\_\_方法。大致思路为：根据所赋值的类别（Parameter, Module或Tensor），进行不同的操作。对于Parameter和Module对象，将其分别放入self._parameters和self._modules中的OrderedDict末端。对于Tensor,则放入self._buffers或者新建一个属性，并赋值。详细分析见代码：
```python
    def __setattr__(self, name, value):
        def remove_from(*dicts):
            for d in dicts:
                if name in d:
                    del d[name]
        # 首先判断value是不是Parameter
        # 取出现有的Parameter
        params = self.__dict__.get('_parameters')
        # value是Parameter
        if isinstance(value, Parameter):
            if params is None:
                raise AttributeError(
                    "cannot assign parameters before Module.__init__() call")
            # 为避免命名冲突，删除同名的item
            remove_from(self.__dict__, self._buffers, self._modules)
            # 注册这个Parameter到self._parameters中，详见下文
            self.register_parameter(name, value)
        # 如果value不是Parameter，并且现有的Parameter已存在name
        # 则value必须是None
        # 这段代码用于清除self._parameters中特定name的Parameter(赋为None)
        elif params is not None and name in params:
            if value is not None:
                raise TypeError("cannot assign '{}' as parameter '{}' "
                                "(torch.nn.Parameter or None expected)"
                                .format(torch.typename(value), name))
            self.register_parameter(name, value)
        else: 
        # value为Module,与前一种情况类似
            modules = self.__dict__.get('_modules')
            if isinstance(value, Module):
                if modules is None:
                    raise AttributeError(
                        "cannot assign module before Module.__init__() call")
                remove_from(self.__dict__, self._parameters, self._buffers)
                modules[name] = value # 此处完成注册工作
            elif modules is not None and name in modules:
                if value is not None:
                    raise TypeError("cannot assign '{}' as child module '{}' "
                                    "(torch.nn.Module or None expected)"
                                    .format(torch.typename(value), name))
                modules[name] = value # 此处完成清除工作
            else:
            # value既不是Parameter也不是Module
                buffers = self.__dict__.get('_buffers')
                # 如果name已存在于self._buffers中，
                if buffers is not None and name in buffers:
                    # 此时，value必须为Tensor或者None
                    if value is not None and not torch.is_tensor(value):
                        raise TypeError("cannot assign '{}' as buffer '{}' "
                                        "(torch.Tensor or None expected)"
                                        .format(torch.typename(value), name))
                    buffers[name] = value # 注册(更准确的说：更新)到self._buffers中
                else:
                    # 不属于上述任何一种情况，则开辟新属性，并赋值
                    object.__setattr__(self, name, value)
```
注意到注册Parameter时用到了self.register_parameter方法，与注册Modules与Buffers略有不同：
```python
    def register_parameter(self, name, param):
    # 一系列的检查
        """Adds a parameter to the module.

        The parameter can be accessed as an attribute using given name.
        """
        if '_parameters' not in self.__dict__:
            raise AttributeError(
                "cannot assign parameter before Module.__init__() call")
        if param is None:
            self._parameters[name] = None
        elif not isinstance(param, Parameter):
            raise TypeError("cannot assign '{}' object to parameter '{}' "
                            "(torch.nn.Parameter or None required)"
                            .format(torch.typename(param), name))
        elif param.grad_fn:
            # Parameter必须是一个leaf对象，creator=None
            raise ValueError(
                "Cannot assign non-leaf Variable to parameter '{0}'. Model "
                "parameters must be created explicitly. To express '{0}' "
                "as a function of another variable, compute the value in "
                "the forward() method.".format(name))
        else:
            self._parameters[name] = param
```            
纵观整个赋值流程，我们发现如果所赋的值为Parameter，则无需再次调用self.register_parameter方法。在我之前写的*[Center loss代码](https://github.com/jxgu1016/MNIST_center_loss.pytorch)* 中，我就错误的将centers这个Param注册了2次。
```python
    def __init__(self, num_classes, feat_dim, loss_weight=1.0):
        super(CenterLoss, self).__init__()
        self.num_classes = num_classes
        self.feat_dim = feat_dim
        self.loss_weight = loss_weight
        self.centers = nn.Parameter(torch.randn(num_classes, feat_dim))
        # no need to register manually. See nn.Module.__setattr__(...)
        # self.register_parameter('centers', self.centers) 
```   
## forward与\__call\__
基类中没有实现forward方法，需要在子类中进一步明确。实际调用forward时，也无需直接使用forward，而是通过\__call\__这个魔法方法来更为简洁直观地实现。另外一个好处就是，为各种hook提供了空间(否则hook就需要写到forward里，这样就不美观了)。
```python
    output = Net(input) # 而不是 output = Net.forward(input)
```
源码如下：
```python
    def __call__(self, *input, **kwargs):
        # 对input进行预处理
        for hook in self._forward_pre_hooks.values():
            hook(self, input)
        # 调用forward
        result = self.forward(*input, **kwargs)
        # 另外一个hook处理
        for hook in self._forward_hooks.values():
            hook_result = hook(self, input, result)
            if hook_result is not None:
                raise RuntimeError(
                    "forward hooks should never return any values, but '{}'"
                    "didn't return None".format(hook))
        # 反传hook
        if len(self._backward_hooks) > 0:
            var = result
            while not isinstance(var, Variable):
                var = var[0]
            grad_fn = var.grad_fn
            if grad_fn is not None:
                for hook in self._backward_hooks.values():
                    wrapper = functools.partial(hook, self)
                    functools.update_wrapper(wrapper, hook)
                    grad_fn.register_hook(wrapper)
        return result
```       
## parameters生成器
正如在[另外一篇笔记](http://leanote.com/blog/post/5993a7f6ab64411d390007a2)中所说，self.parameters返回一个包含这一Module中所有Parameter的生成器，供optim中的算法使用。代码挺简单：
```python
    def parameters(self):
        """Returns an iterator over module parameters.

        This is typically passed to an optimizer.

        Example:
            >>> for param in model.parameters():
            >>>     print(type(param.data), param.size())
            <class 'torch.FloatTensor'> (20L,)
            <class 'torch.FloatTensor'> (20L, 1L, 5L, 5L)
        """
        for name, param in self.named_parameters():
            yield param

    def named_parameters(self, memo=None, prefix=''):
        """Returns an iterator over module parameters, yielding both the
        name of the parameter as well as the parameter itself

        Example:
            >>> for name, param in self.named_parameters():
            >>>    if name in ['bias']:
            >>>        print(param.size())
        """
        if memo is None:
            memo = set()
        for name, p in self._parameters.items():
            # 根据param来判断是否重复注册，同一个param即使有两个注册名也不行
            if p is not None and p not in memo:
                memo.add(p)
                yield prefix + ('.' if prefix else '') + name, p
        # 子Module中的params也能访问到，且命名也体现了其所处的子类名称
        for mname, module in self.named_children():
            submodule_prefix = prefix + ('.' if prefix else '') + mname
            for name, p in module.named_parameters(memo, submodule_prefix):
                yield name, p
```              
## children与modules方法
两者的区别主要在于，前者返回的是当前Module的**直系子代** Module所构成的生成器，而后者返回当前Module的所有子代Module **连同自身**所构成的生成器。
前者代码：
```python
    def children(self):
        """Returns an iterator over immediate children modules."""
        for name, module in self.named_children():
            yield module

    def named_children(self):
        """Returns an iterator over immediate children modules, yielding both
        the name of the module as well as the module itself."""
        
        memo = set()
        # 仅搜索了直系子代
        for name, module in self._modules.items():
            if module is not None and module not in memo:
                memo.add(module)
                yield name, module
```               
后者代码：
```python
    def modules(self):
        """Returns an iterator over all modules in the network."""
        for name, module in self.named_modules():
            yield module

    def named_modules(self, memo=None, prefix=''):
        """Returns an iterator over all modules in the network, yielding
        both the name of the module as well as the module itself."""
        if memo is None:
            memo = set()
        if self not in memo:
            memo.add(self)
            yield prefix, self
            for name, module in self._modules.items():
                if module is None:
                    continue
                submodule_prefix = prefix + ('.' if prefix else '') + name
                # 迭代遍历搜索
                for m in module.named_modules(memo, submodule_prefix):
                    yield m
```
nn.Module中hook需要一篇笔记专门分析，而剩余的方法对于理解这个类没有核心作用，故省略。
