import numpy as np
x = np.array([[-1,-1, 1],
             [-1,1,1],
             [1,-1,1],
             [1,1,1]
             ])
y = np.array([-1,-1,-1,1])
w=np.zeros(x.shape[1])

def hebbian(x,y):
    global w
    for i in range(len(x)):
        delta_weight=x[i]*y[i]
        w=w+delta_weight

hebbian(x,y)
def test_net(x,w):
    for i in range(len(x)):
        output =np.sign(np.dot(x[i],w))
        print(f"Input: {x[i][:2]},Output:{output}")
print("weights after training",w)
print("Testing the network")
test_net(x,w)




