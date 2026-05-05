
# UGTrack 工程规范

## 基本要求

1.先读取AGENTS.md和UGTrack.md再进行后续操作
2.使用UTF-8读取文件内容
3.运行指令前需要先激活ostrack环境：conda activate ostrack
4.输出内容规范：文档/说明中文，代码英文，注释双语
5.对于每一个聊天窗口，在vibe_coding生成对应标题的文件夹，文件夹内创建：进度.md、问题及解决措施.md、修改文件.md
6.冒烟测试后删除相关脚本和生成文件
7.训练时根据output\logs估算训练完成时间
8.生成的文件遵守不污染原来ostrack文件，而是在ugtrack/下新建重写，这样既可以保留原有内容还可以添加新功能
9.修改lib\config\ugtrack\config.py后，更新doc\UGTrack\YAML说明.md
10.数据集相关的内容参考data\OTB100_UWB\OTB100_UWB说明.md、data\OTB100_UWB\README.md、lib\train\dataset\otb100_uwb.py
11.训练完成后删除多余的pth.tar，只留下最后一轮的pth.tar

