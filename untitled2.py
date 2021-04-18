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

movies=["Annie Hall","Ben-Hur","Casablance","Gandhi","West Side Story"]
num_oscars=[5,11,3,8,10]
#plot bars with left x_coordinates [0,1,2,3,4],heights [num_oscars]
plt.bar(range(len(movies)),num_oscars)
#add a title
plt.title("My Favorite Movies")
#label the y-axis
plt.ylabel("# of Academy Awards")
#label x-axis with movie names at bar centers
#plt.xticks(range(len(movies)),movies) #相当于x轴
plt.show()