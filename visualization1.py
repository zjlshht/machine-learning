import matplotlib.pyplot as plt
import numpy as np
import matplotlib
matplotlib.rcParams['backend'] = "Qt4Agg"
x=np.linspace(0,20)
plt.plot(x, 5+x)
plt.plot(x, 1+2*x,'--')#虚线作图
plt.show()
plt.scatter(x, 5+x, c=200*x,label="scatter plot")
plt.legend(loc='upper left')#给标签定位
plt.grid()#上网格
plt.xlabel('x')
plt.ylabel('y')
plt.title('x or y')
plt.show()
#plt.semilogy(x, 3**x+2,'0')

