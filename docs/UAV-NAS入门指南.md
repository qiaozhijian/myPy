> 管理员: 刘珮泽(pliuan@connect.ust.hk), 徐洋(), 李昊佳()，吴易易

当前服务都在墙外所以内地访问可能会有corner case 没有解决，可以找管理员看看怎么办



WebDav 连接主要是方便在外网查看数据 (数据在云端)

SMB 连接是内网高速访问数据(数据在云端)

Synology Drive 是类似 Dropbox的工具(本地和云端同步)



## 设立账号
1. 浏览器访问 https://dsm.hkust-uav.online/ 查看是否能够看到界面, 账号请找 徐洋,刘珮泽,李昊佳开号。
2. 开号后在邮件箱根据连接重置密码
3. 登录网页查看空间是否已经完成初始化，画面如下 
	![image.png](https://khalil-picgo-1321910894.cos.ap-hongkong.myqcloud.com/images/202405272357358.png)



## Webdav (广域网络硬盘) 

1. Webdav 可以作为 Zotero 的数据库存储
2. 在广域网访问大文件的时候不太好使

#### Windows
找到共享空间中的 raidrive-2019-12-22-ads-free.exe 安装到本地。
点击添加-》找到NAS-》WebDAV

![image.png](https://khalil-picgo-1321910894.cos.ap-hongkong.myqcloud.com/images/202405280011483.png)
使用 webdav.hkust-uav.online 域名 443 端口 ； 填写自己的账户和密码即可
![image.png](https://khalil-picgo-1321910894.cos.ap-hongkong.myqcloud.com/images/202405280015629.png)
点击，弹出文件夹表示设置完成。
![image.png](https://khalil-picgo-1321910894.cos.ap-hongkong.myqcloud.com/images/202405280016488.png)
现在就能看到这里有一个网络硬盘啦

#### MacOS



#### Ubuntu

![image-20240528124837271](https://khalil-picgo-1321910894.cos.ap-hongkong.myqcloud.com/images/202405281248455.png)

连接地址为： dav://webdav.hkust-uav.online

输入自己的账号和密码就可以了



## SMB (局域网网络硬盘)

> 提供在局域网内的快速访问和存储

### Windows

在网络下访问 UAV-Group 这个节点，登录自己的账号即可

![image-20240528103136237](https://khalil-picgo-1321910894.cos.ap-hongkong.myqcloud.com/images/202405281031286.png)

### MacOS

在网络下访问 UAV-Group 这个节点，登录自己的账号即可

### Ubuntu

在网络下访问 UAV-Group 这个节点，登录自己的账号即可

![image-20240528124007379](https://khalil-picgo-1321910894.cos.ap-hongkong.myqcloud.com/images/202405281240671.png)





## 私有DropBox

因为PC端走的端口和协议不同，所以需要配置代理才能正常使用。

需要先下载 Synology Drive Client

代理工具下载: [cloudflared](https://developers.cloudflare.com/cloudflare-one/connections/connect-networks/downloads/)

### Windows

##### 开启代理

进入 `ShareSpace\Synology_Driver\Win` 

`Synology_Driver_Win.vbs`: 一键启动端口代理

`Kill_Synology_Driver.cmd`: 一键干掉端口代理

##### 将开启代理脚本加入自启动（Optional）

`Synology_Driver_TaskSche.xml`: 计划任务程序模板，建议把`Synology_Driver_Win.vbs`设置为开机启动，导入计划任务后记得更改路径为本地的脚本路径。

这一步只是为了更方便，但是因为代理占用了本地的6690端口，所以会导致可能会有端口冲突，所以手动开启会更方便debug。

##### 配置服务 （必须要开启代理）

![](https://khalil-picgo-1321910894.cos.ap-hongkong.myqcloud.com/images/202405281106153.png)

在登录界面使用 127.0.0.1连接到NAS， 不要使用 Quick connect!!!!! (使用Quick connect 在广域网访问会非常慢) 一通傻瓜操作过后就可以像 Dropbox 一样使用了。（优点就是量大管饱，缺点就是可能上下行会有一些限制）



### MacOS
##### 开启代理
##### 配置服务



### Linux
##### 开启代理

查看cloudflared 是否安装好

命令为 

```sh
cloudflared access tcp --hostname drive.hkust-uav.online --url localhost:6690
```

![image-20240528140652671](https://khalil-picgo-1321910894.cos.ap-hongkong.myqcloud.com/images/202405281406728.png)

出现一下就说明代理成功。

##### 配置服务

和windows 平台一样，使用127.0.0.1 连接即可

##### 开机自动代理

1. 打开终端,编辑 systemd 服务文件

   ```shell
   sudo vim /etc/systemd/system/cloudflared.service
   ```

2. 在文件中添加以下内容:

   ```shell
   [Unit]
   Description=Cloudflared TCP
   After=network-online.target
   Wants=network-online.target
   
   [Service]
   ExecStart=/usr/local/bin/cloudflared access tcp --hostname drive.hkust-uav.online --url localhost:6690
   Restart=always
   User=root
   
   [Install]
   WantedBy=multi-user.target
   ```

3. 保存退出

4. 启用并启动服务:

   ```shell
   sudo systemctl enable cloudflared.service
   sudo systemctl start cloudflared.service
   ```

5. 查看是否正常启动

   ```shell
   sudo systemctl status cloudflared.service
   ```

   ![image-20240528141958253](https://khalil-picgo-1321910894.cos.ap-hongkong.myqcloud.com/images/202405281419342.png)



## 移动设备访问(手机 Ipad 等)

> 移动设备就比较方便了

需要下载 Synology Drive

地址: dsm.hkust-uav.online

输入用户名 和 密码 就能在内外网正常访问





