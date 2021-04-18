'''折线图'''
from matplotlib import pyplot as plt
years=[1950,1960,1970,1980,1990,2000,2010]
gdp=[300.2,543.3,1075.9,2862.5,5979.6,10289.7,14958.3]
#create a line chart,years on x-axis,gdp on y-axis
plt.plot(years,gdp,color='green',marker='o',linestyle='solid')
# add a title
plt.title("Nominal GDP")
#add a label to the y_axis
plt.ylabel("Billions of $")
plt.show()
'''条形图'''
movies=["Annie Hall","Ben-Hur","Casablance","Gandhi","West Side Story"]
num_oscars=[5,11,3,8,10]
#plot bars with left x_coordinates [0,1,2,3,4],heights [num_oscars]
plt.bar(range(len(movies)),num_oscars)
#add a title
plt.title("My Favorite Movies")
#label the y-axis
plt.ylabel("# of Academy Awards")
#label x-axis with movie names at bar centers
plt.xticks(range(len(movies)),movies) #相当于x轴
plt.show()
'''特殊条形图'''
from collections import Counter
grades=[83,95,91,87,70,0,85,82,100,67,73,77,0]
#Bucket grades by decile,but put 100 in with the 90s
histogram=Counter(min(grade//10*10,90)for grade in grades)#核心 90是最后一类 不太懂

plt.bar([x+5 for x in histogram.keys()],#shift bars right by 5 相应横坐标
        histogram.values(), #give each bar a its correct height 相应纵坐标
        10,                 #give each bar a width of 10 条形宽度
        edgecolor=(0,0,0))  #Black edges for each bar 边缘色

plt.axis([-5,105,0,5])      #x-axis from -5 to 105
                            #y-axis from 0 to 
plt.xticks([10*i for i in range(11)]) #x-axis labels at 0,10,...,100
plt.xlabel("Decile")
plt.ylabel("# of students")
plt.title("Distribution of Exam 1 Grades")
plt.show()

mentions=[500,505]
years=[2017,2018]
plt.bar(years,mentions,0.8)
plt.xticks(years)
plt.ylabel("# of times I heard someone say'data science'")
#if you don't do this,matplotlib will label the x-axis 0,1
#and then add a +2.013e3 off in the corner (bad matplotlib!)
plt.ticklabel_format(useOffset=False) #这个不知道干什么的 删了结果也一样
#misleading y-axis only shows the part above 500
plt.axis([2016.5,2018.5,499,506])
plt.title("Look at the'Huge' Increase!")
plt.show()
'''趋势图'''
variance=[1,2,4,8,16,32,64,128,256]
bias_squared=[256,128,64,32,16,8,4,2,1]
total_error=[x+y for x,y in zip(variance,bias_squared)]#用zip产生对应组合
xs=[ i for i, _ in enumerate(variance)]#产生和variance维数一样的自然数数组
#we can make multiple calls to plt.plot
#to show multiple series on the same chart
plt.plot(xs,variance, 'g-',label='varience')#green solid line
plt.plot(xs,bias_squared,'r-.',label='bias^2')#red dot-dashed line
plt.plot(xs,total_error, 'b:',label='total error')#bule dotted line

#because we've assigned labels to each series
#we can get a legend for free(loc=9 means "top center")
plt.legend(loc=9)
plt.xlabel("model complexity")
plt.xticks([])#剔除横坐标
plt.title("The Bias-variance Tradeoff")
plt.show()
'''散点图'''
friends=[70,65,72,63,71,64,60,64,67]
'''demo of enumerate
a=[ i for i,_ in enumerate(friends) if i%2==0]
a=[ x for i,x in enumerate(friends) if i%2==0]
'''
minutes=[175,170,205,120,220,130,105,145,190]
labels=['a','b','c','d','e','f','g','h','i']
plt.scatter(friends,minutes)
#label each point
for label,friend_count,minute_count in zip(labels,friends,minutes):#逐个画点标签
    plt.annotate(label,
                 xy=(friend_count,minute_count),#Put the label with its point 点坐标
                 xytext=(5,-5),#but slightly offset 标签相对点的坐标（右下方）
                 textcoords='offset points')
plt.title("Daily Minutes vs. Number of Friends")
plt.xlabel("# of friends")
plt.ylabel("daily minutes spent on the site")
plt.show()
'''概率密度函数图'''
from function import normal_pdf,normal_cdf,binomial
xs=[x/10.0 for x in range(-50,50)]
plt.plot(xs,[normal_pdf(x,sigma=1) for x in xs],'-',label='mu=0,sigma=1')
plt.plot(xs,[normal_pdf(x,sigma=0.5) for x in xs],':',label='mu=0,sigma=0.5')
plt.plot(xs,[normal_pdf(x,mu=-1) for x in xs],'-.',label='mu=-1,sigma=1')
plt.legend()#显示标签
plt.title("Various Normal pdfs")
plt.show()
'''概率分布函数图'''
xs=[x/10.0 for x in range(-50,50)]
plt.plot(xs,[normal_cdf(x,sigma=1) for x in xs],'-',label='mu=0,sigma=1')
plt.plot(xs,[normal_cdf(x,sigma=2) for x in xs],'--',label='mu=0,sigma=2')
plt.plot(xs,[normal_cdf(x,sigma=0.5) for x in xs],':',label='mu=0,sigma=0.5')
plt.plot(xs,[normal_cdf(x,mu=-1) for x in xs],'-.',label='mu=-1,sigma=1')
plt.legend(loc=4)#bottom right
plt.title("Various Normal cdfs")
plt.show()
'''中心极限定理'''
import math
def binomial_histogram(p:float,n:list,num_points:int)->None:
    '''Picks points from  a Binomial(n,p) and plots their histogram'''
    data = [binomial(n, p) for _ in range(num_points)]    
# use a bar chart to show the actual binomial samples 
    histogram=Counter(data)
    plt.bar([x-0.4 for x in histogram.keys()],
            [v / num_points for v in histogram.values()],
            0.8,
            color='0.75')
    mu=p*n
    sigma=math.sqrt(n*p*(1-p))
    #use a line chart to show the normal approximation 
    xs=range(min(data),max(data)+1)
    ys=[normal_cdf(i+0.5,mu,sigma)-normal_cdf(i-0.5,mu,sigma) for i in xs]
    plt.plot(xs,ys)
    plt.title("Binomial Distribution vs. Normal Approximation")
    plt.show()
    
binomial_histogram(0.4,500,10000)


