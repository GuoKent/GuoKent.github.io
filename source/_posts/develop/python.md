---
title: Python
date: 2024-12-31 00:00:00
tags:
- python
- 开发
categories:
- 开发笔记
alias:
- developnotes/python/
---

# 装饰器@property

装饰器作用就是把方法method转换为属性property。因此被@property装饰的成员函数，**只能有一个参数self；不能和别的类属性同名；并且在调用时不需要加()。**

> 换句话说就是把函数变成一个属性(数值)

如果只有@property装饰，那么value是**只读不可写**的。因此在property装饰的基础上，还附赠了@x.setter装饰器和@x.deleter装饰器。
```python
class A():    
    def __init__(self):
        self._value = 1 
        
    @property
    def value(self): 
        return self._value
    
    @value.setter
    def value(self, x)
        if x <= 0:
           raise ValueError('value must > 0')
        self._value = x
    
    @value.deleter
    def value(self):
        del self._value

a = A()
a.value = -1
del a.value # 调用@value.deleter修饰的函数
```

在对 a.value 赋值时，实际上调用的是被@value.setter装饰的函数，我们可以在该函数进行判断数据类型、数据范围等。至此@property装饰适合下面这些场景：

1. **只读不可修改的属性**。只需要实现@property
2. 输入对 setter **进行判断**。
3. 需要**实时地计算**属性值。

解释一下第三种情况，比如我们已知电阻阻值和电压，要求电流，最好的方式就是实现@property装饰的函数，可以像属性一样访问电流，并且是实时计算的。
```python
class OHM():    
    def __init__(self):
        self._U = 1
        self._R = 1

    @property
    def I(self):
        return self._U / self._R

    @property
    def R(self):
        return self._R

    @R.setter
    def R(self,r):
        if r <= 0:
            raise ValueError('r must >0')
        self._R = r

ohm = OHM()
ohm.R = 1
print(ohm.I)
```

# 类方法@classmethod

**@classmethod** 修饰符对应的函数不需要实例化，不需要 `self` 参数，但第一个参数需要是表示自身类的 `cls` 参数，可以来调用类的属性，类的方法，实例化对象等。
```python
class A(object):
    bar = 1
    
    def func1(self):  
        print ('foo') 
    
    @classmethod
    def func2(cls):
        print ('func2')
        print (cls.bar)
        cls().func1()   # 调用 foo 方法
 
A.func2()               # 不需要实例化
```

`@classmethod` 的作用实际是可以在 class 内实例化 class ，一般使用在有工厂模式要求时。作用就是比如输入的数据需要清洗一遍再实例化，可以把清洗函数定义在 class 内部并加上 @classmethod 装饰器已达到减少代码的目的。

总结起来就是：@classmethod可以用来为一个类创建一些**预处理的实例**。

## 举例
```python
class Data_test(object):
    day=0
    month=0
    year=0
    def __init__(self, year = 0, month = 0, day = 0):
        self.day = day
        self.month = month
        self.year = year
    def out_date(self):
        print("year :", self.year)
        print("month :", self.month)
        print("day :", self.day)
t = Data_test(2020,1,1)
t.out_date()
```
> 一个普通的类调用方法

如果用户输入的是 "2016-8-1" 这样的字符格式，那么就需要调用Date_test 类前做一下处理
```python
class Data_test2(object):
    day = 0
    month = 0
    year = 0
    def __init__(self,year=0,month=0,day=0):
        self.day = day
        self.month = month
        self.year = year
    @classmethod
    def get_date(cls, data_as_string):
 
        #这里第一个参数是cls， 表示调用当前的类名
        year, month, day = map(int, data_as_string.split('-'))
        date1 = cls(year, month, day)     #返回的是一个初始化后的类
        return date1
 
    def out_date(self):
        print("year :", self.year)
        print("month :", self.month)
        print("day :", self.day)
    
r = Data_test2.get_date("2020-1-1")
r.out_date()
```
> 采用@classmethod进行预处理

这样子等于先调用 get_date() 对字符串进行出来，然后才使用 Data_test 的构造函数初始化。这样的好处就是你以后重构类的时候不必要修改构造函数，只需要额外添加你要处理的函数，然后使用装饰符 @classmethod 就可以了。

# 传参collate_fn
在使用 dataloader 时，常需要对每个 batch 进行预处理，比如清理异常值、预处理文本等等。在大模型计算时，文本常常需要使用 processor 进行处理，而这一步一般在模型外部计算，因此可以放入 collate_fn 来加速处理
```python
# 自定义dataset
class MyDataset(Dataset):
	def __init__(self):
		super().__init__()
		
	def __getitem__(self, index):
		return x, y
		
# 无参collate_fn
def collate_fn(batch):
		codes, url, txt_inputs, img_inputs = zip(*batch)
    inputs = self.processor(
        text=list(txt_inputs),
        images=list(img_inputs),
        videos=None,
        padding=True,
        return_tensors="pt",
    )
  return list(codes), list(url), inputs

# 有参collate_fn, 以类的方式定义批处理函数
class Collate:
    def __init__(self, processor):
        self.processor = processor # processor为自定义传入的参数

    def __call__(self, batch):
        codes, url, txt_inputs, img_inputs = zip(*batch)
        inputs = self.processor(
            text=list(txt_inputs),
            images=list(img_inputs),
            videos=None,
            padding=True,
            return_tensors="pt",
        )
        return list(codes), list(url), inputs
 
 # 使用
 def prepare_dataset(processor, args):
    dataset = MyDataset(args.data_path, args.sys_prompt, processor)
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        pin_memory=True,
        num_workers=args.num_workers,
        collate_fn=Collate(processor)  # 传入调用函数的对象
    )
    return dataset, dataloader
```