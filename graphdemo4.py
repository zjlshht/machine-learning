from function import random_normal,correlation
from matplotlib import pyplot as plt
'''相关系数图'''
xs=[random_normal() for _ in range(1000)]
ys1=[x+random_normal()/2 for x in xs]
ys2=[-x+random_normal()/2 for x in xs]
plt.scatter(xs,ys1,marker='.',color='black',label='ys1')
plt.scatter(xs,ys2,marker='.',color='gray',label='ys2')
plt.xlabel('xs')
plt.ylabel('ys')
plt.legend(loc=9)
plt.title("Very Different Joint Distributions")
plt.show()
print(correlation(xs,ys1))
print(correlation(xs,ys2))
'''相关矩阵图'''
a=[[1,1,1,2,2,2],[2,2,2,3,3,3],[-1,-1,-1,-2,-2,-2]] #任意一个向量列表
num_vectors=len(a)
fig,ax=plt.subplots(num_vectors,num_vectors)
for i in range(num_vectors):
    for j in range(num_vectors):
    #Scatter column_j on the x-axis vs. column_i on the y-axis
        if i !=j:
            ax[i][j].scatter(a[j],a[i])
    #unless i==j,in which case show the series name
        else:
            ax[i][j].annotate("series"+str(i),(0.5,0.5),xycoords='axes fraction',
                              ha="center",va="center")
    #Then hide axis labels except left and bottom charts
        if i<num_vectors -1:
            ax[i][j].xaxis.set_visible(False)
        if j>0:
            ax[i][j].yaxis.set_visible(False)
    #Fix the bottom-right and top-left axis labels,which are wrong because
    #their charts only have text in them
    ax[-1][-1].set_xlim(ax[0][-1].get_xlim())
    ax[0][0].set_ylim(ax[0][1].get_ylim())
    plt.show()
        

