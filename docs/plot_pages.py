import datetime
import matplotlib.pyplot as plt

# make up some data
x = []
y = []
with open("pages.txt" , "r") as f:
    line = f.readline()
    while line:
        line = line.strip()
        tstr = line[0:19]
        print(line[26:])
        y.append(int(line[26:]))
        tdatetime = datetime.datetime.strptime(tstr, '%Y-%m-%d %H:%M:%S')
        x.append(tdatetime)
        line = f.readline()

# plot
plt.plot(x,y)
# beautify the x-labels
plt.gcf().autofmt_xdate()
plt.ylabel("page number")

plt.savefig("pages.png")
