# Know How Than Know Why

---



## 网络连接的三种方式

1. 桥接模式：虚拟系统可以和外部系统通讯，但是容易造成IP冲突
2. NAT模式：网络地址转换模式，虚拟系统可以和外部系统通讯，不造成IP冲突
3. 主机模式：独立的系统





## Linux目录结构

基本介绍：

1. linux的文件系统是采用层级式的树状目录结构，在此结构中的最上层是根目录“/”，然后在此目录下再创建其他的目录
2. **Linux世界里，一切皆文件**

- 具体的目录结构

  - /bin [**常用**] (/usr/bin、/usr/local/bin)，binary的缩写，这个目录存放着最经常使用的命令
  - /sbin (/usr/sbin,/usr/local/sbin) s是Super User的意思，这里存放的是系统管理员使用的系统管理程序。
  - /home [**常用**] 存放普通用户的主目录，在Linux中每个用户都有一个自己的目录，一般该目录名是以用户的账号命名
  - /root [**常用**] 该目录是系统管理员，也称超级权限者的用户主目录
  - /lib 系统开机所需要最基本的动态链接共享库，其作用类似于Windows里的DLL文件。几乎所有的应用程序都需要用到这些共享库
  - /lost+found 这个目录一般情况下是空的，当系统非法关机后，这里就存放了一些文件
  - /etc [**常用**] 所有的系统管理所需要的配置文件和子目录，比如安装mysql数据库
  - /usr [**常用**] 这是一个非常重要的目录，用户的很多应用程序和文件都放在这个目录下，类似于windows下的program files 目录
  - /boot [**常用**] 存放的是启动Linux时使用的一些核心文件，包括一些链接文件以及镜像文件
  - /proc [勿动] 这个目录是一个虚拟的目录，它是系统内存的映射，访问这个目录来获取系统信息
  - /srv [勿动] service 缩写，该目录存放一些服务启动之后需要提取的数据
  - /sys [勿动] 该目录下安装了2.6内核中新出现的一个文件系统sysfs
  - /tmp 这个目录是用来存放一些临时文件
  - /dev 类似于windows的设备管理器，把所有的硬件用文件的形式存储
  - /media [**常用**] linux系统会自动识别一些设备，识别后，linux会把识别的设备挂载到这个目录下
  - /mnt [**常用**] 系统提供该目录是为了让用户临时挂载别的文件系统的，我们可以将外部的存储挂载在/mnt/ 上，然后进入该目录就可以查看里面的内容了。
  - /opt 给主机额外**安装软件**所摆放的目录。 默认为空。
  - /usr/local [**常用**] 这是另一个给主机额外安装软件所安装的目录。一般是通过编译源码的方式安装的程序
  - /var [**常用**] 该目录存放着在不断扩充着的东西，习惯将经常被修改的目录放在这个目录下。包括各种日志文件
  - /selinux [security-enhanced linux] SELinux 是一种安全子系统，它能控制程序只能访问特定文件，有三种工作模式，可以自行设置

  

  


## 实操篇



### vi和vim的基本介绍

Linux系统会内置vi文本编辑器

Vim具有程序编辑的能力，可以看作是Vi的增强版本，可以主动的以字体颜色辨别语法的正确性，方便程序设计。代码补完、编译即错误跳转等方便编程的功能特别丰富，在程序员中被广泛使用。

#### vi和vim常用的三种模式

- 正常模式：以vim打开一个档案就直接进入一般模式了（默认模式）。在这个模式中，你可以使用「上下左右」按键来移动光标，你可以使用「删除字符」或「删除整行」来处理档案内容，也可以使用「复制、粘贴」来处理你的文件数据。
- 插入模式：按下「i」键进入编辑模式
- 命令行模式：**输入esc再输入:**  进入这个模式，该模式中可以提供相关指令，完成读取、存盘、替换、离开vim、显示行号等动作(w=write，q=quit)

---

常用快捷键

1. 在一般模式下，拷贝当前行「yy」，拷贝当前行向下的5行 「5yy」，粘贴输入「p」
2. 在一般模式下，删除当前行 「dd」，删除当前行向下的5行「5dd」
3. 在文件中查找某个单词，在命令行下输入「/关键字」，按下回车查找，输入「n」就是查找下一个
4. 设置文件的行号，取消文件的行号。在命令行下「:set nu」，「:set nonu」
5. 在一般模式下，使用快捷键到该文档的最末行「G」和最首行「gg」
6. 撤销动作是在一般模式下输入「u」
7. 在一般模式下，将光标移动到指定行，输入「 行号 在输入 shift+g」

---

### 关机 & 重启

```sh
shutdown -h now # 立刻关机
shutdown -h 1 # 1分钟后关机
shutdown -r now #现在重新启动
halt # 关机
reboot # 重启
sync # 把内存的数据同步到磁盘，关机或重启之前先执行sync命令
```

### 用户登录和注销

- 基本介绍

1. 登陆时尽量少用root登录，因为它是系统管理员，拥有最大权限，为了避免操作失误经量少用。利用普通用户登录后，使用“su - 用户名”可以切换成系统管理员身份
2. 在提示符下输入logout即可注销账户

- 使用细节

1.  logout注销指令在图形运行级别无效，在运行级别3下有效。



### 用户管理

基本介绍：linux系统是一个多用户多任务的操作系统，任何一个要使用系统资源的用户，都必须首先向系统管理员申请一个账号，然后以这个账号的身份进入系统

#### 添加用户

- 基本语法

```shell
useradd 用户名
例如
useradd wujian # 默认该用户的家目录在/home/wujian
```

也可以通过‘useradd -d 指定目录 新的用户名’，给新创建的用户指定家目录

#### 指定/修改密码

- 基本语法

```shell
passwd 用户名 # 修改谁的密码，一定要指定用户名，否则会修改当前用户密码
pwd # 显示当前用户所在的目录
```

#### 删除用户

- 基本语法

```
userdel 用户名 # 删除用户但保留家目录
userdel -r 用户名 #删除用户及用户主目录，此操作要慎重，一般情况下建议保留
```

#### 查询用户信息指令

- 基本语法

```
id 用户名
```

- 应用实例

```
案例：查询root 信息
id root
```

![image-20230921092744902](https://gitee.com/Jerry-wu-jian/img-load/raw/master/noteimg/202309210927970.png)

- 细节说明

当用户不存在时，返回无此用户

#### 切换用户

- 在操作linux中，如果当前用户的权限不够，可以通过`su - 用户名` 指令，切换到高权限用户，比如root。

- 基本语法

```
su - 切换用户名
例如切换到jack
su - jack
```

- 细节说明
  1. 从权限高的用户切换到权限低的用户，不需要输入密码，反之需要。
  2. 党需要返回到原来用户时，使用`exit/logout `指令

#### 查看当前用户/登录用户

```
who am i / whoami
```

#### 用户组

- 类似于角色，系统可以对有共性/权限的多个用户进行统一的管理
- 基本语法

```
# 新增组
groupadd 组名

# 删除组
groupdel 组名

# 增加用户是直接加上组
useradd -g 用户组 用户名
# 例如 增加一个用户zwj，直接将他指定到wudang
useradd -g wudang zwj
# 如果没有指定组，会创建一个同名组，并将用户放入这个组中
```

#### 修改用户组

```
usermod -g 指定用户组 用户名
usermod -g mojiao wudang
# 将zwj切换到mojiao组
```

#### 用户和组相关文件

- /etc/passwd 文件

​	用户（user）的配置文件，记录用户的各种信息

​	每行的含义：<span ><font color = “blue">用户名：口令：用户标识号：组表示号：注释行描述：主目录：登录shell</font></span>

- /etc/shadow 文件

  - 口令的配置文件

  每行的含义：<span><font color="blue">登录名：加密口令：最后一次修改时间：最小时间间隔：最大时间间隔：警告时间：不活动时间：失效时间：标志</font></span>

- /etc/group 文件

  组（group）的配置文件，记录Lnux包含的组的信息

  每行含义：<span><font color="blue">组名：口令：组标识号：组内用户列表</font></span>

### 运行级别

- 基本介绍

运行级别说明：

0：关机

1：单用户【找回丢失密码】

2：多用户状态没有网络服务

3：多用户状态有网络服务

4：系统未使用保留给用户

5：图形界面

6：系统重启

常用运行级别时3和5，也可以指定默认运行级别

- 应用实例

```
init [0123456]
# 应用案例： 通过intit 来切换不同的运行级别，比如 5->3，然后关机
```

#### 指定运行级别

- centos7以前，运行级别定义在 /etc/inittab文件中

- centos7以后进行简化，如下：

  <span><font color="red">multi-user.target</font></span>:analogous to runlevel 3

  <span><font color='red'>graphicl.target</font></span>:analogous to runlevel 5

-  查看当前默认运行级别：

  `systemctl get-default`

- 设置默认运行级别：

  ` systemctl set-default TARGET.target`

  例如` systemctl set-default multi-user.target `

####  找回root密码

1. 首先，启动系统，进入开机界面，在界面中按`e`进入编辑界面

![image-20230921101809247](https://gitee.com/Jerry-wu-jian/img-load/raw/master/noteimg/202309211018309.png)

2. 进入编辑界面，使用键盘上的上下键把光标往下移动，找到以`Linux16 `开头内容所在的行数，在行的最后面输入：`init=/bin/sh`，输入完成后按快捷键：`Ctrl+X`进入单用户模式

![image-20230921101938365](https://gitee.com/Jerry-wu-jian/img-load/raw/master/noteimg/202309211019407.png)

3. 接着，在光标闪烁的位置中输入：`mount -o remount,rw /`(注意：各个单词间有空格，remount，rw之间没有空格)，完成后按键盘的回车键`enter`

![image-20230921102147292](https://gitee.com/Jerry-wu-jian/img-load/raw/master/noteimg/202309211021333.png)

4. 在新的一行后面输入：`passwd`，完成后按键盘的回车键`enter`。输入密码，然后再次确认密码即可（<span><font color='red'>密码长度最好8位以上，但不是必须的</font></span>），密码修改成功后，会显示`passwd.....`的样式，说明密码修改成功.
5. 接着，在鼠标闪烁的位置输入：`touch /.autorelabel`(注意：touch与/有一个空格)，完成后按键盘的回车键`enter`
6. 继续在光标闪烁的位置中，输入:`exec /sbin/init`(注意：exec与/有一个空格)，完成后按键盘的回车键`enter`,等待系统自动修改密码(<span><font color='red'>这个过程时间可能有点长，耐心等待</font></span>)，完成后，系统会自动重启，新的密码生效。

### man/help 帮助指令

- ` man`获得帮助信息

  基本语法 :`man 命令或配置文件` (功能描述：获得帮助信息)

  案例：查看ls命令的帮助信息

  `man ls`

- ` help ` 指令

  基本语法：`help 命令`（功能描述：获得shell内置命令的帮助信息）

  应用实例：查看cd命令的帮助信息

  `help cd`

  

### 文件目录类指令

#### pwd 指令

`pwd`

功能描述：显示当前工作目录的绝对路径

#### ls 指令

`ls [选项] [目录或者文件]`

常用选项：

-a：显示当前目录所有的文件和目录，包括隐藏的

-l：以列表的方式显示信息

-lh:以常见单位显示大小

#### cd 指令

`cd [参数]` 

功能描述：切换到指定目录

`cd ~`or`cd`：回到自己的家目录

`cd ..`:回到当前目录的上一级目录

#### mkdir 指令

mkdir 指令用于创建目录

`mkdir [选项] 要创建的目录`

常用选项：

-p：创建多级目录

应用实例:

```
案例1：创建一个目录 /home/dog
mkdir /home/dog
案例2：创建多级目录 /home/animal/tiger,因为没有animal文件，所以一次创建两个文件要加-p
mkdir -p /home/animal/tiger
```

#### rmdir指令

rmdir指令删除空目录

`rmdir [选项] [要删除的空目录]`

应用实例：删除一个目录/home/dog

`rmdir /home/dog`

rmdir删除的是空目录，如果目录下有内容时时无法删除的。

若要删除非空目录，需要使用`rm -rf 要删除的目录`(慎用)

#### touch 指令

touch指令创建空文件

`touch 文件名称`

案例：创建一个空文件hello.txt

`touch hello.txt`s



#### cp 指令

cp 指令拷贝文件到指定目录

`cp [选项] 需要拷贝的文件 目的地文件目录`

常用选项：

-r：递归复制整个文件夹

应用实例：

``` 
案例1： 将/home/hello.txt 拷贝到 /home/bbb目录下
mkdir /home/bbb
cp /home/hello.txt /home/bbb
案例2： 递归复制整个文件夹,比如将/home/bbb整个目录拷贝到opt
cp -r /home/bbb/ /opt/
使用细节：强制覆盖不提示的方法：\cp
```

#### rm 指令

说明：rm指令移除文件或目录

`rm [选项] 要删除的文件或目录`

常用选项：

-r：递归删除整个文件夹

-f：强制删除不提示

应用实例：

```
案例1： 将/home/hello.txt 删除
rm /home/hello.txt
案例2: 递归删除整个文件夹/home/bbb
rm -r /home/bbb(删除有提示) or rm -rf /home/bbb(删除不提示)
使用细节：强制删除不提示的方法：带上-f参数即可
```

#### mv 指令

说明：移动文件与目录或重命名

`mv oldNameFile newNameFile`(功能描述：重命名)

`mv /temp/movefile /targetFolder`(功能描述：移动文件)

```
案例1： 将/home/cat.txt 文件重新命名为 pig.txt
mv /home/cat.txt pig.txt
案例2： 将/home/pig.txt 文件移动到 /root目录下
mv /home/pig.txt /root/
案例3: 移动整个目录,比如将/opt/bbb 移动到 /home, m
mv /opt/bbb/ /home/
```

#### cat 指令

cat 查看文件内容

`cat [选项] 要查看的文件`

常用选项：

-n:显示行号

应用实例:

案例1： 查看/etc/profile 文件内容，并显示行号

`cat -n /etc/profile`

使用细节：cat 只能浏览文件，不能修改文件，为了浏览方便，一般会带上管道命令`| more`

`cat -n /etc/profile | more` 一行一行看按enter，按页看按空格

管道命令有点类似于将上一个命令交给下一个命令处理。

 

#### more 指令

more指令时一个基于vi编辑器的文本过滤器，它以全屏幕的方式按页显示文本文件的内容。more指令中内置了若干快捷键（交互的指令），详见操作说明

`more 要查看的文件`

![image-20230921154046197](https://gitee.com/Jerry-wu-jian/img-load/raw/master/noteimg/202309211540277.png)

#### less 指令

less 指令用来分屏查看文件内容，它的功能与more指令类似，但是比more指令更加强大，支持各种显示终端。less指令在显示文件内容时，并不是一次将整个文件加载之后才显示，而是根据显示需要加载内容，对于显示大型文件具有较高的效率。

`less 要查看的文件`

![image-20230921154608829](https://gitee.com/Jerry-wu-jian/img-load/raw/master/noteimg/202309211546891.png)

#### echo 指令

echo 输出内容到控制台

`echo 选项 输出内容`

应用实例

```
案例：使用echo 指令输出环境变量,比如输出 $PATH $HOSTNAME
echo $PATH echo $HOSTNAME
案例：使用echo 指令输出hello，world！
echo "hello,world!"
```

#### head 指令

用于显示文件的开头部分内容，默认情况下head指令显示文件的前10行内容

`head 文件` (功能描述：查看文件头10行内容)

`head -n 5 文件 ` (功能描述： 查看文件头5行内容，5可以是任意行数)

应用实例：查看 /etc/profile 的前面5行代码

`head -n 5 /etc/profile`



#### tail 指令

用于输出文件中尾部的内容，默认情况下tail指令显示文件的前10行内容。

`tail 文件` 功能描述：查看文件尾10行内容

`tail -n 5 文件`功能描述：查看文件尾5行内容，5可以是任意行数

`tail -f 文件` 功能描述：实时追踪该文档的所有更新

应用实例

```
案例1：查看/etc/profile 最后5行代码
tail -n /etc/profile
案例2：实时监控mydate.txt，查看文件有变化时，是否看到，实时的追加日期
taile -f mydate.txt
```



#### > 指令 和>>指令

`>`输出重定向(覆盖写)和`>>`追加

```
基本语法：
ls -l > 文件 (功能描述：列表的内容写入文件中（覆盖写）)
ls -al >> 文件 (功能描述：列表的内容追加到文件的末尾)
cat 文件1 > 文件2 (功能描述：将文件1的内容覆盖到文件2)
echo "内容" >> 文件 (将“内容”追加到文件中)
```

#### ln 指令

软链接也成为符号链接，类似于windows里的快捷方式，主要存放了链接其他文件的路径。

```
ln -s [源文件或目录] [软链接名] (功能描述：给原文件创建一个软连接)
应用实例：
案例1：在/home目录下创建一个软链接myroot，链接到/root目录
ln -s /root /home/myroot
案例2： 删除软链接myroot
rm -f /home/myroot
细节说明，当我们使用pwd指令查看目录时，仍然看到的时软链接所在目录。
```

#### history 指令

查看已经执行过历史命令，也可以执行历史命令

```
基本语法：
history 
应用实例
案例1：显示所有的历史命令
history
案例2：显示最近使用过的10个指令
history 10
案例3：执行历史编号为5的指令
!5
```

### 时间日期类指令

#### date指令-显示当前日期/设置日期

基本语法

1. `date` （功能描述：显示当前时间）
2. `date +%Y` （功能描述：显示当前年份）
3. `date +%m` (功能描述：显示当前月份)
4. `date +%d` (功能描述：显示当前是哪一天)
5. `date “+%Y-%m-%d %H:%M:%S`(功能描述：显示年月日时分秒)
6. `date -s 字符串时间`

应用实例：

设置系统当前时间：比如设置成 2023-11-11  11：11：11

`date -s "2023-11-11 11:11:11"` 

#### cal 指令 显示日历

基本语法

`cal [option]`（功能描述：不加选项，显示本月日历）

案例1：显示当前日历

`cal` 

案例2：显示2020年日历

`cal 2020`

### 搜索查找类指令

#### find 指令

find指令将从**指定目录向下递归地遍历其各个子目录**，将满足条件的文件或者目录显示在终端

基本语法：

`find [搜索范围] [option]`

选项说明：

![image-20231003195625708](https://gitee.com/Jerry-wu-jian/img-load/raw/master/noteimg/202310031956802.png)

```
应用实例
案例1：按文件名：根据名称查找/home 目录下的hello.txt文件
find /home -name hello.txt
案例2：按拥有者：查找/opt目录下，用户名称为nobody的文件
find /opt -user nobody
案例3：查找整个linux系统下大于200M的文件（+ 大于 -小于 等于,k;M;G）
find / -size +200M
```

#### locate 指令

locate指令可以快速定位文件路径。locate指令利用事先建立的系统中的所有文件名称以及路径的locate数据库实现快速定位，locate 指令无需遍历整个文件系统，查询速度快。为u了保证查询结果的准确度，管理员必须定期更新locate时刻。

基本语法：

`locate 待搜索文件`

特别说明：

由于locate指令基于数据库进行查询，所以第一次运行前，必须使用updatedb指令创建locate 数据库

应用实例：

案例1：请使用locate指令快速定位hello.txt文件所在目录

`updatedb` `locate hello.txt`

#### which 指令

which指令可以查看某个指令在哪个目录下，比如ls指令在哪个目录

`which ls`

#### grep 号令和管道符号|

grep过滤查找，管道符号“|”，表示将前一个命令的处理结果输出传递给后面的命令处理

基本语法：

`grep [option] [查找内容] [源文件]`

常用选项：

-n：显示匹配行及行号

-i：忽略字母大小写

应用案例：

案例1：请在hello.txt文件中，查找“yes”所在行，并且显示行号

写法1：`cat /home/hello.txt | grep -n "yes" `

写法2：`grep -n "yes" /home/hello.txt`

### 压缩和解压类指令

#### gzip/gunzip 指令

gzip用于压缩文件，gunzip用于解压文件

基本语法

`gzip [文件]` 功能描述：压缩文件，只能将文件压缩为`*.gz`文件

`gunzip [文件.gz]` 功能描述：解压缩文件命令

```
应用实例：
案例1：将/home下的hello.txt文件进行压缩
gzip /home/hello.txt
案例2： 将/home下的 hello.txt.gz文件解压缩
gunzip /home/hello.txt.gz
```



#### zip/unzip 指令

zip用于压缩文件，unzip用于解压，在项目打包发布中很有用

基本语法：

`zip [option] [文件.zip]`

`unzip [option] [file.zip]`

常用选项：

-r：递归压缩，即压缩目录

unzip常用选项：

-d  <目录>  ：指定解压后文件的存放目录

```
应用实例：
案例1：将/home下的所有文件进行压缩成myhome.zip
zip -r myhome.zip /home/
案例2：将myhome.zip解压到/opt/tmp目录下
mkdir /opt/tmp
unzip -d /opt/tmp /home/myhome.zip
```

#### tar 指令

tar指令是打包指令，最后打包的文件时.tar.gz的文件。既可以压缩也可以解压

基本语法：`tar [option] xxx.tar.gz 打包的内容`

选项说明：

-c ：产生.tar打包文件

-v：显示详细信息

-f：指定压缩后的文件名

-z：打包同时压缩

-x：解包.tar文件

```
应用实例：
案例1：压缩多个文件，将/home/pig.txt和/home/cat.txt压缩成pc.tar.gz
tar -zcvf pc.tar.gz /home/pig.txt /home/cat.txt
案例2：将/home的文件夹压缩成myhome.tar.gz
tar -zcvf myhome.tar.gz /home/
案例3：将 pc.tar.gz解压到当前目录，切换到/opt/
tar -x
案例4：将myhome.tar.gz解压到/opt/tmp2目录下
mkdir /opt/tmp2
tar -zxvf /home/myhome.tar.gz -C /opt/tmp2
-C代表change dir
```

### 组管理和权限管理

在linux重的每个用户必须属于一个组，不能独立于组外，在linux中每个文件有所有者、所在组、其他组的概念

#### 文件/目录 所有者

一般为文件的创建者，谁创建了该文件，就自然的成为该文件的所有者

##### 查看文件的所有者指令

`ls -ahl`





##### 修改文件所有者指令

`chown [userid] [filename]`

案例：使用root创建一个文件apple.txt 然后将其所有者修改成tom

`touch apple.txt`

`chown tom apple.txt`

#### 组的创建

基本指令

`groupadd [groupname]`

应用实例：创建一个组monster，创建一个用户fox，并放入到monster组中

`groupadd monster`,`useradd -g monster fox`



#### 文件/目录所在组

当某个用户创建了一个文件后，这个文件的所在组就是该用户所在的组

##### 修改文件所在的组

基本指令

`chgrp [groupname] [filename]`

应用实例：使用root用户创建文件orange.txt，看看当前这个文件属于哪个组，然后将这个文件所在组，修改到fruit组

`ls -ahl`

`chgrp fruit orange.txt`

#### 改变用户所在组

在添加用户时，可以指定将该用户添加到哪个组中，同样的用root的管理权限可以改变某个用户所在的组

改变用户所在的组

`usermod -g [groupname] [username]`

`usermod -d [directory name] [username] [改变该用户登录的初始目录]`<span><font color="red">特别说明：</font></span>用户需要有进入到新目录的权限

应用实例：将zwj这个用户从原来所在组，修改到wudang组

`usermod -g wudang zwj`

#### 权限的基本介绍

ls -l中现实的内容如下：

`-rwxrw-r-- 1 root root 1213 datetime`

0-9位说明

1. 第0位确定文件类型(d,-,l,c,b)
   l是链接，相当于快捷方式
   d是目录，相当于文件夹
   c是**字符设备**文件，鼠标，键盘之类
   b是块设备，比如硬盘
2. 第1-3为确定所有者（该文件的所有者）拥有该文件的权限，——user
3. 第4-6位确定所属组（同用户组的）拥有该文件的权限，——group
4. 第7-9位确定其他用户拥有该文件的权限，——other

##### rwx作用到文件

1. [r]代表可读read，可以读取，查看
2. [w]代表可写write，可以修改，但是不代表可以删除该文件
3. [x]代表可以执行execute：可以被执行。

##### rwx作用到目录

1. [r]代表可读read，ls查看目录内容
2. [w]可以修改write，对目录内创建+删除+重命名
3. [x]代表可以执行execute：可以进入该目录

##### 修改权限-chmod

通过chmod指令，可以修改文件或者目录的权限

第一种方式：+，-，=变更权限

u：所有者，g：所有组，o：其他人 ，a：所有人（ugo的总和）

1)`chmod u=rwx,g=rx,o=x [filename/dir name]` 直接修改权限

2)`chmod o+w [filename/dir name]` 给其他人添加写权限

3)`chmod a-x [filename/ dir name]` 给所有人去掉执行权限

```
案例演示：
1)给abc文件的所有者读写执行的权限，给所在组读写执行权限，给其他组读执行权限
chmod u=rwx,g=rx,o=rx abc
2)给abc文件的所有者除去执行的权限，增加组写的权限
chmod u-x,g+w abc
3)给abc文件的所有用户添加读的权限
chmod a+r abc
```

第二种方式：通过数字变更权限

r=4,w=2,x=1

`chmod u=rwx,g=rx,o=x [filename/dir name]`与`chmod 751 [filename/dir name]`相同

##### 修改文件所有者-chown

基本介绍：

`chown newowner [filename   /dir name] `功能描述：改变所有者

`chown newowner:newgroup [filename/dir] `功能描述：改变所有者和所在组

-R 如果是目录，则使其下所有子文件或目录递归生效

```
案例演示
请将/home/abc.txt文件所有者改成tom
chown tom /home/abc.txt 
请将/home/kkk 目录下的所有的文件和目录时的所有者都修改成tom
chmod -R tom /home/kkk/ 
```

##### 修改文件/目录所在组-chgrp

基本介绍：`chgrp newgroup [filename/dir name]` 功能描述：改变所在组

```
案例演示
1.请将/home/abc.txt文件的所在组修改成shaolin
groupadd shaolin
chgrp shaolin /home/abc.txt
2.请将/home/text目录下所有的文件和目录的所在组都修改成shaolin
chgrp -R shaolin /home/text
```

### 定时任务调度

#### crond 任务调度

crontab 进行定时任务的设置

##### 概述

任务调度：指系统在某个时间执行的特定的命令或程序。

任务调度分类：

- 系统工作：某些重要的工作必须周而复始地执行。如病毒扫描
- 个别用户工作：个别用户可能希望执行某些程序，比如对mysql数据库的备份

示意图：

![image-20231006164632212](https://gitee.com/Jerry-wu-jian/img-load/raw/master/noteimg/202310061646566.png)

基本语法：

`crontab [option]`

常用选项：

-e：编辑crontab定时任务

-l：查询crontab任务

-r：删除当前用户所有的crontab任务

service crond restart [重启任务调度]

##### 快速入门

设置任务调度文件：/etc/crontab

设置个人任务调度：执行`crontab -e`命令

接着输入任务到调度文件：如 `*/1 * * * * ls-l /etc/ > /tmp/to.txt`

意思是每小时的每分钟执行`ls-l /etc/ > /tmp/to.txt`命令

参数细节说明：

5个占位符的说明

![image-20231006165003197](https://gitee.com/Jerry-wu-jian/img-load/raw/master/noteimg/202310061650287.png)

特殊符号的说明：

![image-20231006165024694](https://gitee.com/Jerry-wu-jian/img-load/raw/master/noteimg/202310061650774.png)

特殊时间执行案例：

![image-20231006165043184](https://gitee.com/Jerry-wu-jian/img-load/raw/master/noteimg/202310061650266.png)

##### crond任务调度实例

![image-20231030085316109](https://gitee.com/Jerry-wu-jian/img-load/raw/master/noteimg/202310300853208.png)

```
应用实例
案例1：每隔1分钟，就将当日的日期信息，追加到/tmp/mydate文件中
*/1 * * * * date >> /tmp/mydate

案例2：每隔1分钟，将当前日期和日历都追加到/home/mycal文件中
*/1 * * * * date >> /home/mycal
*/1 * * * * cal >> /home/mycal
或者写一个shell脚本定时执行脚本
(1)vim /home/my.sh 写入date >> /home/mycal  cal >> /home/mycal
(2)给执行权限chmod u+x /home/my.sh
(3)crontab -e 增加 */1 * * * * /home/my.sh

案例3：每天凌晨2：00将mysql数据库testdb，被分到文件中。提示：指令为mysqldump -u root -p密码 数据库 >> /home/db.bak
crontab -e
0 2 * * * mysqldump -u root -proot testdb >> /home/db.bak
```

#### at定时任务

基本介绍：

1. at命令是一次性定时计划任务，at的守护进程atd会以后台模式运行，检查作业队列来运行。

2. 默认情况下，atd守护进程每60s检查作业队列，有作业时，会检查作业运行时间，如果时间与当前时间匹配，则运行次作业。
3. at命令是一次性定时计划任务，执行完一个任务后不再执行此任务了
4. 在使用at命令的时候，一定要保证atd进程的启动，可以使用相关指令来查看，`ps -ef | grep atd`



at命令格式

`at [option] [time]` Ctrl+D 结束at命令的输入

![image-20231030100607637](https://gitee.com/Jerry-wu-jian/img-load/raw/master/noteimg/202310301006720.png)

##### at时间定义

at指定时间的方法：

1)接受在当天的hh:mm(hour:minute)式的时间指定。假如该时间已过去，那么就放在第二天执行。

2)使用midnight，noon，teatime等比较模糊的词语来指定时间。

3)采用12小时计时制，即在时间后面加上AM或PM来说明是上午还是下午，例如8pm。

4)指定命令执行的具体日期，指定格式为 month day（月 日）或mm/dd/yy（月/日/年）或dd.mm.yy（日.月.年），指定的日期必须跟在指定时间的后面。例如：04:00 2021-03-1

5)使用相对计时法。指定格式为：now + count time-units， now就是当前时间，timu-units是时间单位，这里能够是minutes、hours、days、weeks。count是时间的数量，几天，几小时。例如：now + 5 minutes。

6)直接使用today，tomorrow来指定完成命令的时间。

##### 应用实例

```
案例1：2天后的下午5点执行/bin/ls /home
at 5pm +  days
/bin/ls /home 输入ctrl+d
```

![image-20231030104758911](https://gitee.com/Jerry-wu-jian/img-load/raw/master/noteimg/202310301047999.png)

```
案例2：atq命令查看系统中没有执行的工作任务
案例3：明天17殿中，输出时间到指定文件内，比如/root/date100.log
at 5pm tomorrow
date > /root/date100.log

案例4：2分钟后，输出时间到指定文件内，比如/root/date200.log
at now + 2 minutes
date > /root/date200.log

案例5：删除已经设置的任务，atrm 编号
atrm 4 // 表示将job队列中编号为4的jobs
```

### 显示系统执行的进程

- 基本介绍

  ps命令用来查看目前系统中，有哪些正在执行，以及它们执行的状况，可以不加任何参数

  `ps -a`:显示当前终端的所有进程信息

  `ps -u`:以用户的格式显示进程i西南西

  `ps -x`:显示后台进程运行的参数

  `ps -ef`:以全格式显示当前所有进程，-e显示所有进程，-f以全格式

  

![image-20240103100146273](https://gitee.com/Jerry-wu-jian/img-load/raw/master/noteimg/202401031001581.png)
