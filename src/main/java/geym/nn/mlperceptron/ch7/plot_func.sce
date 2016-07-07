clf();
i=4;
t = read('C:/x.txt', 1, 100);
t1 = read('C:/y.txt', 1, 100);

x=[-2:0.04:2]
plot(t,t1,'k+',x,sin(%pi*i/4*x)+1,'k');

