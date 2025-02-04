---
title: Pytorch 分布式训练
date: 2024-10-01 00:00:00
tags:
- 分布式
- 开发
- Pytorch
- Python
categories:
- 开发笔记
alias:
- developnotes/distribute/
---
## 手动配置分布式训练
该方法自定义程度化较高，但环境等需自己配置，代码写起来较繁杂
### 准备环境
```python
import torch
from torch.nn.parallel import DistributedDataParallel as DDP

def setup_distributed(rank, world_size):
    '''
    初始化分布式环境
    '''
    os.environ["MASTER_ADDR"] = "127.0.0.1"  # 本机地址
    os.environ["MASTER_PORT"] = "29946"  # 端口号，任取一个空端口就行
    dist.init_process_group(
        backend='nccl',  # NCCL 是 GPU 上分布式训练的推荐后端
        init_method='env://',  # 使用环境变量初始化
        rank=rank,
        world_size=world_size
    )
    torch.cuda.set_device(rank)  # 将当前进程绑定到 rank 对应的 GPU

def prepare_model(model, rank):
    '''
    将 torch.model 放入分布式模型中
    '''
    # model = model.to(rank)  # 将模型移动到 rank 对应的 GPU
    executor = model._nn_executor.model.to(rank)  # 将模型中torch.model部分放入gpu
    ddp_model = DDP(executor, device_ids=[rank])  # 使用 DDP 包装模型(torch.model类)
    return ddp_model

def prepare_data(dataset, rank, world_size, batch_size, collate_fn=None, num_workers=0):
    '''
    数据并行, 数据转为分布式数据
    '''
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=False)
    dataloader = DataLoader(dataset, 
                            batch_size=batch_size, 
                            sampler=sampler, 
                            collate_fn=collate_fn, 
                            num_workers=num_workers,
                            pin_memory=True,
                            prefetch_factor=8,
                            persistent_workers=True)
    return dataloader
```
- `num_workers` 线程数，一般去cpu线程数的 1/2，或取gpu数量。但取多了会占大量内存
- `pin_memory` 固定数据在内存中的地址，可加快读取速度，但可能会导致占用内存大
- `prefetch_factor` 预先取多少个batch到内存中，默认为2，调大可加快读取速度
- `persistent_workers` 每次迭代结束是否保留进程，默认为False，可加快读写速度
- `collate_fn` 默认将 `[(data 1, label 1), (data 2, label 2), …]` 转化为`[[data 1, data 2, ...], [label 1, label 2, ...]]` 若要自定义`collate_fn` 则需自行转换

### 执行函数
```python
def main(model, dataset):
    '''
    主运行函数 (主进程)
    '''
    world_size = torch.cuda.device_count()
    mp.spawn(
        inference,  # 传入推理/训练函数, 默认会把第一个rank参数传入
        args=(world_size, model, dataset),  # 推理/训练函数的其他参数
        nprocs=world_size,
        join=True
    )
    ...
 
 def inference(rank, world_size, model, dataset):
    '''
    推理/训练函数 (每个gpu执行的函数)
    Args:
        rank: 当前 gpu 对应的 rank
        world_size: gpu 总数
        model: torch.model
        dataset: torch.dataset
    '''
    # 初始化分布式环境
    setup_distributed(rank, world_size)
    # 准备模型
    ddp_model: DDP = prepare_model(model, rank)
    # 准备数据
    dataloader = prepare_data(dataset, rank, world_size, batch_size=BATCH_SIZE, collate_fn=None, num_workers=NUM_WORKERS)
    # 推理
    ddp_model.eval()
    fail_batch = 0
    print(f"Begin inference, model rank: {rank}")
    with torch.no_grad():
        for batch in tqdm(dataloader, total=len(dataloader)):
```

### 结果写入和保存
由于是多卡推理/训练，涉及到文件读写冲突问题，因此需要制定策略防止文件写冲突
- 每张卡各自写到自己的文件，整个训练/推理过程结束完最后再合并（推荐，并行写入更快）
- 只有一个结果文件，每张卡轮流写入（进程写入结果文件要排队，降低效率）
```python
import fcntl

def write_result_to_file(batch, results, rank):
	''' 每个线程的结果写入临时文件, 或者单独写入一个文件'''
	sim_temp_path = f"./res/temp/results_rank_{rank}.txt"
    is_header = False if os.path.exists(sim_temp_path) else True
    
    # ... 结果处理，得到写入文件的格式
    new_df = pd.DataFrame(new_rows)  # 要写入文件的格式
    
    # 写入临时文件
    with open(sim_temp_path, 'a') as f:
        # 独占锁
        fcntl.flock(f, fcntl.LOCK_EX)
        try:
            new_df.to_csv(f, sep='\t', index=False, header=is_header, mode='a')
        except Exception as e:
            print(f"Raise Error: {e}")
        finally:
            # 解锁
            fcntl.flock(f, fcntl.LOCK_UN)
    '''
    当 num_workers 设定大于gpu数量时，一个gpu可能会执行多个线程的任务。
    当线程1再cuda:0上执行完，然后执行写入临时文件。若线程1的写文件还没执行完，线程2也在
    cuda:0上执行完，也开始写入临时文件，就会发生冲突
    因此需要一个互斥锁来保证两者的写操作冲突
    '''
 
 def merge_results_from_files(world_size, save_path):
    '''
    将每个gpu的结果合并到一起
    '''
    is_header = False if os.path.exists(save_path) else True
    # 合并每个rank的结果
    for rank in range(world_size):
        sim_temp_path = f"./res/temp/results_rank_{rank}.txt"
        rank_file = pd.read_table(sim_temp_path, sep='\t', encoding="utf-8")
        rank_file.to_csv(save_path, sep='\t', index=False, header=is_header, mode='a')
    print(f"Finish merge file to: {save_path}")

def delete_temp_file():
    '''删除临时文件(可选)'''
    temp_folder = "./res/temp/"
    temp_file_names = [f"results_rank_{rank}.txt" for rank in range(torch.cuda.device_count())]
    for file_name in temp_file_names:
        file_path = os.path.join(temp_folder, file_name)
        if os.path.isfile(file_path):
            try:
                os.remove(file_path)
                print(f"Delete file: {file_path} successfully")
            except Exception as e:
                print(f"Raise Error when delete {file_path}: {e}")
```

## 自动配置分布式训练
另一种分布式训练写法，就是使用torchrun来执行python文件。运行的主函数只需关注每一个gpu的代码怎么运行即可，torchrun会自动分配环境给每一gpu。该方法只需考虑每个 gpu 对应的执行函数即可，代码写起来较为简单，也无需考虑文件互斥的问题，运行时直接 torchrun 自动执行分布式环境
> 一个典型的例子：[CLIP](https://github.com/OFA-Sys/Chinese-CLIP/blob/master/cn_clip/training/main.py)

### 执行函数
```python
import torch.nn.parallel.DistributedDataParallel as DistributedDataParallel

def main():
    '''
    每个 gpu 的执行函数
    '''
    args = parse_args()
    
    # 查看当前gpu是哪个rank
    args.local_device_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(args.local_device_rank)
    args.device = torch.device("cuda", args.local_device_rank)

    print("Init process group...")  # 准备环境
    dist.init_process_group(backend="nccl", init_method='env://')
    args.rank = dist.get_rank()
    args.world_size = dist.get_world_size()
    
    # 准备模型
    model = MyModel()  # 定义自己的模型
    model = DistributedDataParallel(model, ...)  # 放入分布式模型里
    
    # 准备数据集
    dataset = MyDataset()
    sampler = DistributedSampler(dataset, shuffle=True, seed=args.seed)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        pin_memory=False,
        num_workers=args.num_workers if is_train else args.valid_num_workers,
        sampler=sampler,
        collate_fn=collate_fn
    )
    
    # 优化器
    optimizer = optim.AdamW(...)
    
    # 训练
    train(model, ...)
 
if __name__ == "__main__":
		main()
```

### 执行脚本
```shell
# 默认8卡全部
nohup torchrun --nproc_per_node=8 --master_port=29500 train_cnclip.py --max-epochs 10 --use-augment > ./logs/train.log 2>&1 &

# 指定其中几张卡(默认按顺序取)
nohup CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 --master_port=29500 train_cnclip.py --max-epochs 10 --use-augment > ./logs/train.log 2>&1 &

# 默认按顺序取前4张卡
nohup torchrun --nproc_per_node=4 --master_port=29500 train_cnclip.py --max-epochs 10 --use-augment > ./logs/train.log 2>&1 &



```