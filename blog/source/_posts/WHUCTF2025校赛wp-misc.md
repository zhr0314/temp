---
title: WHUCTF2025校赛wp-misc
date: 2025-04-03 02:53:57
tags: 'CTF'
---

## 行星防御理事会（前半）

听音频知SSTV，导入软件得到图像：
![SSTV解析的图像](/images/WHUCTF2025校赛wp-misc/1.png)
图像中有Passwd: 2024-YR4
用此passwd对音频进行解密（deepsound），得到secret.zip，再用此passwd对压缩包解压，得到QR_modified.png和script.py
（后半略）

## [签到]益智游戏

### Part1：数独

直接上求解器，再上行列式计算器
![求解器](/images/WHUCTF2025校赛wp-misc/2.png)
![行列式计算](/images/WHUCTF2025校赛wp-misc/3.png)
25770150*81=2087382150，以此为密码解压缩包得到flag1：
WHUCTF{Little_games_reall

### Part2：数织

同样上求解器
![求解器](/images/WHUCTF2025校赛wp-misc/4.jpg)
然后二进制转十进制得key：（忘记这里的key是多少了），再解包得flag2：
y_train_your_brain_&_play

### Part3：鬼脚图

由于学长算错了key，所以此部分忽略

三部分连起来即得flag：
WHUCTF{Little_games_really_train_your_brain_&_play_more_in_cn.puzzle_website}

## 青轴还是红轴

先用length筛选出带有usbhid.data的条目：
![筛选后的条目](/images/WHUCTF2025校赛wp-misc/5.png)
再将其导出为JSON文件。
然后过脚本，提取出usbhid.data数据：
```python
import json

# 1. 读取JSON文件
input_file = "./2.json"  # 替换为你的JSON文件路径
output_file = "./usbhid_data2.txt"  # 保存提取结果的文件路径

with open(input_file, "r", encoding="utf-8") as f:
    data = json.load(f)  # 加载JSON数据

# 2. 提取所有usbhid.data值
usbhid_data_values = []
for item in data:
    if "_source" in item and "layers" in item["_source"] and "usbhid.data" in item["_source"]["layers"]:
        usbhid_data_values.append(item["_source"]["layers"]["usbhid.data"])

# 3. 保存到新文件
with open(output_file, "w", encoding="utf-8") as f:
    for value in usbhid_data_values:
        f.write(value + "\n")  # 每个值写入一行

print(f"提取完成，结果已保存到 {output_file}")

```
再将usbhid.data数据用脚本转化为键值，进而转换为对应的按键：
```python
#!/usr/bin/env python
# -*- coding:utf-8 -*-
 
normalKeys = {"04":"a", "05":"b", "06":"c", "07":"d", "08":"e", "09":"f", "0a":"g", "0b":"h", "0c":"i", "0d":"j", "0e":"k", "0f":"l", "10":"m", "11":"n", "12":"o", "13":"p", "14":"q", "15":"r", "16":"s", "17":"t", "18":"u", "19":"v", "1a":"w", "1b":"x", "1c":"y", "1d":"z","1e":"1", "1f":"2", "20":"3", "21":"4", "22":"5", "23":"6","24":"7","25":"8","26":"9","27":"0","28":"<RET>","29":"<ESC>","2a":"<DEL>", "2b":"\t","2c":"<SPACE>","2d":"-","2e":"=","2f":"[","30":"]","31":"\\","32":"<NON>","33":";","34":"'","35":"<GA>","36":",","37":".","38":"/","39":"<CAP>","3a":"<F1>","3b":"<F2>", "3c":"<F3>","3d":"<F4>","3e":"<F5>","3f":"<F6>","40":"<F7>","41":"<F8>","42":"<F9>","43":"<F10>","44":"<F11>","45":"<F12>"}
shiftKeys = {"04":"A", "05":"B", "06":"C", "07":"D", "08":"E", "09":"F", "0a":"G", "0b":"H", "0c":"I", "0d":"J", "0e":"K", "0f":"L", "10":"M", "11":"N", "12":"O", "13":"P", "14":"Q", "15":"R", "16":"S", "17":"T", "18":"U", "19":"V", "1a":"W", "1b":"X", "1c":"Y", "1d":"Z","1e":"!", "1f":"@", "20":"#", "21":"$", "22":"%", "23":"^","24":"&","25":"*","26":"(","27":")","28":"<RET>","29":"<ESC>","2a":"<DEL>", "2b":"\t","2c":"<SPACE>","2d":"_","2e":"+","2f":"{","30":"}","31":"|","32":"<NON>","33":"\"","34":":","35":"<GA>","36":"<","37":">","38":"?","39":"<CAP>","3a":"<F1>","3b":"<F2>", "3c":"<F3>","3d":"<F4>","3e":"<F5>","3f":"<F6>","40":"<F7>","41":"<F8>","42":"<F9>","43":"<F10>","44":"<F11>","45":"<F12>"}
output = []
keys = open('atta.txt',encoding='utf-8')#提取出来的usbdata.txt文件
for line in keys:
    try:
        if line[0]!='0' or (line[1]!='0' and line[1]!='2') or line[3]!='0' or line[4]!='0' or line[9]!='0' or line[10]!='0' or line[12]!='0' or line[13]!='0' or line[15]!='0' or line[16]!='0' or line[18]!='0' or line[19]!='0' or line[21]!='0' or line[22]!='0' or line[6:8]=="00":
             continue
        if line[6:8] in normalKeys.keys():
            output += [[normalKeys[line[6:8]]],[shiftKeys[line[6:8]]]][line[1]=='2']
        else:
            output += ['[unknown]']
    except:
        pass
keys.close()
 
flag=0
print("".join(output))
for i in range(len(output)):
    try:
        a=output.index('<DEL>')
        del output[a]
        del output[a-1]
    except:
        pass
for i in range(len(output)):
    try:
        if output[i]=="<CAP>":
            flag+=1
            output.pop(i)
            if flag==2:
                flag=0
        if flag!=0:
            output[i]=output[i].upper()
    except:
        pass
print ('output :' + "".join(output))

```
得到按键信息：
IJNIOKJ TFCFG FGTRDCV IUHBGHJ JKIUHNM YGVGH_IJNJKMN UHBN TGBNJUJM GHYTFVB_FGHYGVB YGBNJU_ [DEL]ESZSD DFRESXC HGVBHUHB_UYHBV DCFVG HBU HJKIJNM UYGBN ESZSDX
（处理时[DEL]与前面的空格抵消）
根据键盘上的按键位置画出轨迹得到flag：whuctf{prefer_blue_to_red_switch}