'''
id:0 lane:4
id:1 lane:3
id:2 lane:5
设置成奖励值大于20上传参数
出现的问题
1.奖励值大于20的上传后一直等待下发参数
2.上传参数的client＞2 给所有client发送参数
3.client端迭代次数不统一
解决方案：
1.使用监听机制时刻监听cloud发送的数据
2.开一个线程单独进行io通信
融合后的权重能应对更多环境 但是和client的记忆库并不匹配 导致模型越训练越糟
因为DQN根据记忆库来修正模型，导致loss爆炸
很多情况是接收权重后的第一次轮巡能获得很高的reward 随后降低
修改记录
1.增加记忆库 增加记忆库的训练batch  4.20 10:57
2.修改cloud的send函数发送条件
3.修改client的receive函数的位置
4.加入一个测试client只接受模型
5.只要client接收不属于它本身的权重 就会造成很高的loss
6.client接受融合好的权重时，虽然会获得很高的loss，但也会获得较高的reward
在降低loss过程中会获得很低的reward，降低loss后会获得很好的reward
7.edge只接受融合的参数 其他两个负责上传参数不接收 reward>25才上传
8.减少记忆库， 减少到50

RMS 60次迭代时 40次时NAN 前期挺好的
    40次迭代时 效果不好 但是loss不高
'''

import socket
import struct
import threading
import numpy as np

clientidlist=[]     #存储clientid列表
clientlist=[]       #存储client通信列表
datalist=[]         #存储client权重列表
rewardlist=[]       #存储client奖励列表

class Client(object):
    def __init__(self, skt):
        self.skt = skt

def process_mess(client):
    id = 0
    while True:
        re_data = client.skt.recv(5904)
        if re_data == b"":
            print("id", id)
            print("clientidlist", clientidlist)
            if id in clientidlist:
                clientidlist.remove(id)
            clientlist.remove(client)
            break
        receive = struct.unpack('1i401d256d80d', re_data)
        id = receive[0]
        if id not in clientidlist:
            clientidlist.append(id)
        reward = receive[1]
        rewardlist[id] = reward
        datalist[id] = np.array(receive[2:])
        print("id", id)
        print("reward", reward)
        print("rewardlist[id]", rewardlist[id])
        print("datalist[id]:ok")
        if len(clientidlist) >= 2:
            send_weight()
            clientidlist.clear()
    client.skt.close()


def send_weight():
    sum = np.zeros(736)
    reward_sum = 0
    for i in range(3):
        reward_sum += rewardlist[i]
    for i in range(3):
        pro = rewardlist[i] / reward_sum
        print("pro:", pro)
        sum += np.dot(datalist[i], pro)
    for s in clientlist:
        print("sending mess to:", s)
        s.skt.send(struct.pack('400d256d80d', *sum))
    # s = clientlist[-1]
    # s.skt.send(struct.pack('400d256d80d', *sum))

def run():
    server = socket.socket()
    server.bind(('localhost', 8080))
    server.listen(3)
    print("waiting connect")
    while True:
        serObj, address = server.accept()
        client = Client(serObj)
        clientlist.append(client)
        t=threading.Thread(target=process_mess, args=(client, ))
        t.start()

if __name__ == '__main__':
    for i in range(3):
        datalist.append(0)
        rewardlist.append(0)
    run()





