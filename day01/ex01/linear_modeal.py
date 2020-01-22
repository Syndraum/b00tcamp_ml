import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from mylinearregression import MyLinearRegression as MyLR

data = pd.read_csv("../resources/are_blue_pills_magics.csv")
Xpill = np.array(data["Micrograms"]).reshape(-1, 1)
Yscore = np.array(data["Score"]).reshape(-1, 1)
linear_model1 = MyLR(np.array([[89.0], [-8]]))
linear_model2 = MyLR(np.array([[89.0], [-6]]))
linear_model1.fit_(Xpill, Yscore, alpha=0.01, n_cycle=10)

Y_model1 = linear_model1.predict_(Xpill)
Y_model2 = linear_model2.predict_(Xpill)

# print(linear_model1.mse_(Yscore, Y_model1))
# print(linear_model2.mse_(Yscore, Y_model2))

def fig1():
    plt.plot(Xpill, Yscore, 'co', label='S true(pills)')
    plt.plot(Xpill, Y_model1, 'X--', color='#00FF00',  label='S predict(pills)')
    plt.ylabel('Space driving score')
    plt.xlabel('Quantity of blue pill (in micrograms)')
    plt.legend(loc='lower center', bbox_to_anchor=(0.3, 1.0), ncol=2, edgecolor=None)
    # plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc='lower left',
    #            ncol=2, mode="expand", borderaxespad=0.)
    plt.grid(True)
    plt.show()


def fig2():
    theta1 = np.linspace(-14, -4, 1000)
    for c in range(6):
        linear_model1 = MyLR(np.array([[89.0], [-8]]))
        cost = np.array([])
        linear_model1.theta[0] = 89 + c
        for i in theta1:
            linear_model1.theta[1] = i
            Y_model1 = linear_model1.predict_(Xpill)
            cost = np.append(cost, linear_model1.mse_(Yscore, Y_model1))
        plt.plot(theta1, cost, color='#050505')
    plt.grid(True)
    plt.show()

fig2()