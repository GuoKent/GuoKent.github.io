---
title: Git 笔记
date: 2024-10-01 00:00:00
tags:
- git
- 开发
categories:
- 开发笔记
alias:
- developnotes/git/
---

### git clone
```shell
git clone：
git clone http://xxxx.git  // 默认master分支
git clone -b <branch_name> http://xxxx.git  // clone指定分支

# clone 特定版本/分支
git clone -branch <branch-name> <repo-address>  // 克隆特定分支
git tag  // 列出全部tag(先clone全部仓库)
git checkout <tag>  // 切换到指定tag(版本)
```

### 连接仓库 remote
```shell
git remote -v  // 查看当前连接
git remote remove origin  // 删除现有连接
git remote add origin 你的远程库地址  // 把本地库与远程库关联
```

### git 提交
```shell
# 第一次提交
git init   // 初始化版本库
git add .   // 添加文件到版本库（只是添加到缓存区），.代表添加文件夹下所有文件
git add -f <路径>  // 添加被忽略文件
git commit -m "first commit" // 把添加的文件提交到版本库，并填写提交备注
git push -u origin master    // 第一次推送时
git push  // 第一次推送后，直接使用该命令即可推送修改

# 后续提交
git add .
git commit -m "message"
git push origin master
git push origin <new-brance>  // master无法访问时用新分支 new-branch
```

### 远程更新代码
```shell
git pull origin
git pull origin <branch>  // 初次, 指定分支
```

### 分支操作
```shell
git branch --list  // 查看分支
git branch -m new-name  // 分支重命名
git checkout <branch-name>  // 切换到分支
git checkout -b <branch-name> // 新建分支
git merge target_branch  // 将目标分支合并到当前分支
```