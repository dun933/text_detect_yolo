import json


PICNUM =20000

def index_format(i):
    result = str(i+1)
    while len(result)<5:
        result = '0' + result
    return result

root = 'C:/Users/bz302/Desktop/ReCTS/ReCTS/img/train_ReCTS_0{}.jpg'##icdr 数据目录
train_labels = []
print("reading json")
for i in range(PICNUM):
    with open('C:/Users/bz302/Desktop/ReCTS/ReCTS/gt_unicode/train_ReCTS_0{}.json'.format(index_format(i))) as f:
        train_labels.append(json.loads(f.read()))
print ("reading finished")



