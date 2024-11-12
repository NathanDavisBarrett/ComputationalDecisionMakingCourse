import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider

fig = plt.figure()
ax = fig.add_subplot(111)

# Adjust the subplots region to leave some space for the sliders and buttons
fig.subplots_adjust(left=0.25, bottom=0.25)

ax.set_aspect("equal")

M = 10

xBounds = (-10,25)
yBounds = (-10,25)

ax.set_xlim(xBounds)
ax.set_ylim(yBounds)
l1Line, = ax.plot(xBounds,yBounds,color="green")
l2Line, = ax.plot(xBounds,yBounds,color="blue")
r1Line, = ax.plot(xBounds,yBounds,color="green")
r2Line, = ax.plot(xBounds,yBounds,color="blue")
t1Line, = ax.plot(xBounds,yBounds,color="green")
t2Line, = ax.plot(xBounds,yBounds,color="blue")
b1Line, = ax.plot(xBounds,yBounds,color="green")
b2Line, = ax.plot(xBounds,yBounds,color="blue")

ax.fill_between([1,3],[2,2],[4,4],facecolor="none",edgecolor="green",hatch="//",zorder=10,label="Y_1 = 0 Bounds")

ax.fill_between([7,10],[6,6],[8,8],facecolor="none",edgecolor="blue",hatch="\\\\",zorder=10,label="Y_1 = 0 Bounds")


ax.legend()
ax.set_xlabel("X_1")
ax.set_ylabel("X_2")

slider_ax  = fig.add_axes([0.25, 0.15, 0.65, 0.03])
slider = Slider(slider_ax, 'y_1', 0, 1, valinit=0)

slider2_ax  = fig.add_axes([0.25, 0.1, 0.65, 0.03])
slider2 = Slider(slider2_ax, 'y_2', 0, 1, valinit=1)

fill = ax.fill_between([0,],[0,],[0,],color="red",label="Feasible Region")


def Update(_,setY1):
    global fill

    if setY1:
        y1 = slider.val

        y2 = 1 - y1

        slider2.eventson = False
        slider2.set_val(y2)
        slider2.eventson = True
    else:
        y2 = slider2.val
        y1 = 1 - y2
        slider.eventson = False
        slider.set_val(y1)
        slider.eventson = True

    l1 = 1 - M * (1 - y1)
    r1 = 3 + M * (1 - y1)

    t1 = 4 + M * (1-y1)
    b1 = 2 - M * (1-y1)

    l2 = 7 - M * (1 - y2)
    r2 = 10 + M * (1 - y2)

    t2 = 8 + M * (1-y2)
    b2 = 6 - M * (1-y2)

    l1Line.set_xdata([l1,l1])
    l2Line.set_xdata([l2,l2])

    r1Line.set_xdata([r1,r1])
    r2Line.set_xdata([r2,r2])

    t1Line.set_ydata([t1,t1])
    t2Line.set_ydata([t2,t2])

    b1Line.set_ydata([b1,b1])
    b2Line.set_ydata([b2,b2])


    l = np.max([l1,l2])
    r = np.min([r1,r2])
    t = np.min([t1,t2])
    b = np.max([b1,b2])

    fill.remove()
    fill = ax.fill_between([l,r],[t,t],[b,b],color="red",label="Feasible Region")

    fig.canvas.draw_idle()

slider.on_changed(lambda x: Update(x,True))
slider2.on_changed(lambda x: Update(x,False))

Update(None,True)



plt.show()

