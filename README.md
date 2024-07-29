### README

##### 创建初始化

```python
$ hexo init <folder>	# 初始化project
$ cd <folder> 
$ npm install					# 安装nodejs配置文件
```

>`hexo init <folder>`本命令相当于执行了以下几步：
>
>1. Git clone [hexo-starter](https://github.com/hexojs/hexo-starter) 和 [hexo-theme-landscape](https://github.com/hexojs/hexo-theme-landscape) 主题到当前目录或指定目录。
>2. 使用 [Yarn 1](https://classic.yarnpkg.com/lang/en/)、[pnpm](https://pnpm.io/zh/) 或 [npm](https://docs.npmjs.com/cli/install) 包管理器下载依赖（如有已安装多个，则列在前面的优先）。 npm 默认随 [Node.js](https://hexo.io/zh-cn/docs/index.html#安装-Node-js) 安装
>
>若出现npm权限问题，再在目标文件夹执行`npm install`指令补全所需包(会新增 `node_modules` 文件夹)

##### 文件描述

```c++
├── _config.landscape.yml：主题配置文件，如果
├── _config.yml：站点配置文件，对站点进行配置。
├── node_modules：用来存储已安装的各类依赖包。
├── package-lock.json：包版本依赖锁文件。
├── package.json：应用数据，Hexo的版本以及相关依赖包的版本等。
├── scaffolds：博客模版文件夹，包含page.md，post.md，draft.md三种。
├── source：资源文件夹，存放静态资源如博客md文件、图片等。
└── themes：主题文件夹，Hexo通过将网站内容与主题组合来生成静态网站。
```



##### Hexo 指令

##### 启动服务器

```python
hexo s  # 本地启动项目, s表示server
hexo server
'''
-p, --port		重设端口
-s, --static		只使用静态文件
-l, --log		启用 logger
'''
```

>启动本地服务器，默认 `http://localhost:4000/`.

##### 新建页面

```python
hexo new [layout] <title>
hexo new page -p about/me "About me"  # 新建 source/about/me.md 文件，标题为"About me"
hexo publish [layout] <filename>  # 发布一个草稿，草稿储存在 source/_drafts
'''
<title>		是必须的
-p, --path		指定路径
-r, --replace		替换当前页面
-s, --slug		指定页面 url
'''
```

> 新建一篇文章。 如果没有设置 `layout` 的话，默认使用 [_config.yml](https://hexo.io/zh-cn/docs/configuration) 中的 `default_layout` 参数代替

##### 生成静态文件

```python
hexo g
hexo generate
'''
-d, --deploy  生成文件后部署
-w, --watch  监视文件变动
-b, --bail  生成过程中显示报错
-f, --force  强制重新生成
-c, --concurrency  最大并行生成文件数，默认无限制
'''
```

##### 部署

```python
# 部署之前要下载 hexo git 相关包，在项目的目录下下载
npm install hexo-deployer-git --save

# 部署指令
hexo d
hexo deploy
'''
-g, --generate	部署前生成
'''
```

##### 查看

```python
hexo list <type>  # 列出所有路径
hexo version  # 展示版本信息
hexo config  # 列出 _config.yml 的配置
hexo config [key] [value]  # 指定密钥和值
```

##### 其他

```python
hexo --safe  # 禁止插件和脚本
hexo --debug  # 将信息输出到终端和 debug.log 文件中
hexo --silent  # 禁用终端输出
hexo --config custom.yml  # 自定义config文件
hexo --config custom.yml, custom2.json  # 合并配置文件
hexo --draft  # 展示草稿帖子
hexo --cwd /path/to/cwd  # 自定义工作路径
```





> 官方配置文档：[Hexo](https://hexo.io/zh-cn/)
>
> github 公钥使用：[link](https://blog.csdn.net/qq_39241986/article/details/120192212)
>
> Refer：[link](https://blog.csdn.net/AI_Green/article/details/121675790)