# -*- coding: utf-8 -*-
"""
Created on Thu Feb 11 14:30:47 2021

@author: zjlsyhhht
"""

# 导入模块
import itchat
import datetime
import time
 
# 发送消息
def send_news():
    itchat.auto_login(hotReload = True)    # 会弹出网页二维码，扫描即可，登录你的微信账号，True保持登录状态~
    friend = itchat.search_friends(name = u'')    # name：TA在你微信上的备注（一定要填哦！）
    friend_name = friend[0]['UserName']    # 获取TA的微信名
    while True:    # 会一直执行哦~
        now = datetime.datetime.now()    # 用datetime模块获取当前时间
        now_str = now.strftime('%Y/%m/%d %H:%M:%S')[11:]    # 截取后11位时间
        print('\r{}'.format(now_str),end = '')
        if now_str in ["00:00:00"]:    # 判断当前时间是否为指定时间（因为是00:00的生日祝福，所以这里写的是00:00:00~ 可自行修改~）
            itchat.send('nihao',toUserName = friend_name)    # 设定发送的消息以及发给谁~
            time.sleep(1)    # 等待一秒
            itchat.send('',toUserName = friend_name)
            time.sleep(1)
            itchat.send('',toUserName = friend_name)
            time.sleep(1)
            itchat.send('',toUserName = friend_name)
            
if __name__ == "__main__":    # 主程序~
    send_news()    #调用函数
    exit()    # 退出窗口
 
 