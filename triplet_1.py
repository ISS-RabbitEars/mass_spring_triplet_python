import numpy as np
import sympy as sp
from sympy.physics.vector import dynamicsymbols
from scipy.integrate import odeint
from matplotlib import pyplot as plt
from matplotlib import animation


def integrate(ic, ti, p):
	m, k, req = p
	ic_list = ic

	sub = {}
	for i in range(3):
		sub[M[i]] = m[i]
		sub[K[i]] = k[i]
		sub[Req[i]] = req[i]
		sub[R[i]] = ic_list[4 * i]
		sub[Rdot[i]] = ic_list[4 * i + 1]
		sub[THETA[i]] = ic_list[4 * i + 2]
		sub[THETAdot[i]] = ic_list[4 * i + 3]

	diff_eqs = []
	for i in range(3):
		diff_eqs.append(ic_list[4 * i + 1])
		diff_eqs.append(A[i].subs(sub))
		diff_eqs.append(ic_list[4 * i + 3])
		diff_eqs.append(ALPHA[i].subs(sub))

	print(ti)

	return diff_eqs


t = sp.Symbol('t')
M = sp.symbols('M0:3')
K = sp.symbols('K0:3')
Req = sp.symbols('Req0:3')
R = dynamicsymbols('R0:3')
THETA = dynamicsymbols('THETA0:3')

Rdot = [i.diff(t, 1) for i in R]
Rddot = np.asarray([i.diff(t, 2) for i in R])
THETAdot = [i.diff(t, 1) for i in THETA]
THETAddot = np.asarray([i.diff(t, 2) for i in THETA])

X = [R[i] * sp.cos(THETA[i]) for i in range(3)]
Y = [R[i] * sp.sin(THETA[i]) for i in range(3)]

dR = np.asarray([sp.sqrt((X[i] - X[j])**2 + (Y[i] - Y[j])**2) for i in range(1,3) for j in range(i)])

Xdot = np.asarray([i.diff(t, 1) for i in X])
Ydot = np.asarray([i.diff(t, 1) for i in Y])

T = sp.simplify(sp.Rational(1, 2) * sum(M * (Xdot**2 + Ydot**2)))

V = sp.simplify(sp.Rational(1, 2) * sum(K * (dR - Req)**2))

L = T - V

dLdR = np.asarray([L.diff(i, 1) for i in R])
dLdRdot = np.asarray([L.diff(i, 1) for i in Rdot])
ddtdLdRdot = np.asarray([i.diff(t, 1) for i in dLdRdot])
dLR = ddtdLdRdot - dLdR

dLdTHETA = np.asarray([L.diff(i, 1) for i in THETA])
dLdTHETAdot = np.asarray([L.diff(i, 1) for i in THETAdot])
ddtdLdTHETAdot = np.asarray([i.diff(t, 1) for i in dLdTHETAdot])
dLTHETA = ddtdLdTHETAdot - dLdTHETA

sol = sp.solve(np.append(dLR,dLTHETA).tolist(), np.append(Rddot,THETAddot).tolist())

A = [sol[i] for i in Rddot]
ALPHA = [sol[i] for i in THETAddot]

#------------------------------------------------------------

m = np.array([1, 1, 1])
k = np.array([25, 25, 25])
req = np.array([2, 2, 2])
ro = np.array([1.5, 3, 1.5])
vo = np.array([0, 0, 0])
thetao = np.array([10, 90, 170]) * np.pi/180
omegao = np.array([30, 30, 30]) * np.pi/180
tf = 1

p = m, k, req

ic = []
for i in range(3):
	ic.append(ro[i])
	ic.append(vo[i])
	ic.append(thetao[i])
	ic.append(omegao[i])

nfps = 30
nframes = tf * nfps
ta = np.linspace(0, tf, nframes)

rth = odeint(integrate, ic, ta, args=(p,))

x = np.asarray([[X[i].subs({R[i]:rth[j,4 * i], THETA[i]:rth[j,4 * i + 2]}) for j in range(nframes)] for i in range(3)],dtype=float)
y = np.asarray([[Y[i].subs({R[i]:rth[j,4 * i], THETA[i]:rth[j,4 * i + 2]}) for j in range(nframes)] for i in range(3)],dtype=float)

ke = np.zeros(nframes)
pe = np.zeros(nframes)
for j in range(nframes):
	sub = {}
	for i in range(3):
		sub[M[i]] = m[i]
		sub[K[i]] = k[i]
		sub[Req[i]] = req[i]
		sub[R[i]] = rth[j, 4 * i]
		sub[Rdot[i]] = rth[j, 4 * i + 1]
		sub[THETA[i]] = rth[j, 4 * i + 2]
		sub[THETAdot[i]] = rth[j, 4 * i + 3]
	ke[j] = T.subs(sub)
	pe[j] = V.subs(sub)
E = ke + pe

#------------------------------------------------------------

xmin, xmax = [x.min(), x.max()]
ymin, ymax = [y.min(), y.max()]

msf = 1/25
drs = np.sqrt((xmax - xmin)**2 + (ymax - ymin)**2)
mr = msf * drs
mra = mr * m / max(m)
mras = np.asarray([mra[i]+mra[j] for i in range(1,3) for j in range(i)])

xmax += 2*max(mra)
xmin -= 2*max(mra)
ymax += 2*max(mra)
ymin -= 2*max(mra)

dr = np.asarray([np.sqrt((x[(i+1)%3] - x[i])**2 + (y[(i+1)%3] - y[i])**2) for i in range(3)])
rmax = np.asarray([max(dr[i]) for i in range(3)])
theta = np.asarray([np.arccos((y[(i+1)%3] - y[i])/dr[i]) for i in range(3)])
nl = np.asarray([np.ceil(rmax[i]/mras[i]) for i in range(3)],dtype=int)
l = np.asarray([(dr[i] - mras[i])/nl[i] for i in range(3)])
h = np.asarray([np.sqrt((mras[i]/2)**2 - (0.5 * l[i])**2) for i in range(3)])
flipa = np.zeros((3,nframes))
flipb = np.zeros((3,nframes))
flipc = np.zeros((3,nframes))
for i in range(3):
	flipa[i] = np.asarray([-1 if x[i][j]>x[(i+1)%3][j] and y[i][j]<y[(i+1)%3][j] else 1 for j in range(nframes)])
	flipb[i] = np.asarray([-1 if x[i][j]<x[(i+1)%3][j] and y[i][j]>y[(i+1)%3][j] else 1 for j in range(nframes)])
	flipc[i] = np.asarray([-1 if x[i][j]<x[(i+1)%3][j] else 1 for j in range(nframes)])
xlo = np.zeros((3,nframes))
ylo = np.zeros((3,nframes))
for i in range(3):
	xlo[i] = x[i] + np.sign((y[(i+1)%3] - y[i]) * flipa[i] * flipb[i]) * mra[i] * np.sin(theta[i])
	ylo[i] = y[i] + mra[i] * np.cos(theta[i])
xl = np.zeros((3,max(nl),nframes))
yl = np.zeros((3,max(nl),nframes))
for i in range(3):
	xl[i][0] = xlo[i] + np.sign((y[(i+1)%3]-y[i])*flipa[i]*flipb[i]) * 0.5 * l[i] * np.sin(theta[i]) - np.sign((y[(i+1)%3]-y[i])*flipa[i]*flipb[i]) * flipc[i] * h[i] * np.sin(np.pi/2 - theta[i])
	yl[i][0] = ylo[i] + 0.5 * l[i] * np.cos(theta[i]) + flipc[i] * h[i] * np.cos(np.pi/2 - theta[i])
	for j in range(1,nl[i]):
		xl[i][j] = xlo[i] + np.sign((y[(i+1)%3]-y[i])*flipa[i]*flipb[i]) * (0.5 + j) * l[i] * np.sin(theta[i]) - np.sign((y[(i+1)%3]-y[i])*flipa[i]*flipb[i]) * flipc[i] * (-1)**j * h[i] * np.sin(np.pi/2 - theta[i])
		yl[i][j] = ylo[i] + (0.5 + j) * l[i] * np.cos(theta[i]) + flipc[i] * (-1)**j * h[i] * np.cos(np.pi/2 - theta[i])
xlf = np.zeros((3,nframes))
ylf = np.zeros((3,nframes))
for i in range(3):
	xlf[i] = x[(i+1)%3] - mra[(i+1)%3] * np.sign((y[(i+1)%3]-y[i])*flipa[i]*flipb[i]) * np.sin(theta[i])
	ylf[i] = y[(i+1)%3] - mra[(i+1)%3] * np.cos(theta[i])

fig, a=plt.subplots()

def run(frame):
	plt.clf()
	plt.subplot(211)
	for i in range(3):
		circle=plt.Circle((x[i][frame],y[i][frame]),radius=mra[i],fc='xkcd:red')
		plt.gca().add_patch(circle)
		plt.plot([xlo[i][frame],xl[i][0][frame]],[ylo[i][frame],yl[i][0][frame]],'xkcd:cerulean')
		plt.plot([xl[i][nl[i]-1][frame],xlf[i][frame]],[yl[i][nl[i]-1][frame],ylf[i][frame]],'xkcd:cerulean')
		for j in range(nl[i]-1):
			plt.plot([xl[i][j][frame],xl[i][j+1][frame]],[yl[i][j][frame],yl[i][j+1][frame]],'xkcd:cerulean')
	plt.title("Mass-Spring Triplet")
	ax=plt.gca()
	ax.set_aspect(1)
	plt.xlim([float(xmin),float(xmax)])
	plt.ylim([float(ymin),float(ymax)])
	ax.xaxis.set_ticklabels([])
	ax.yaxis.set_ticklabels([])
	ax.xaxis.set_ticks_position('none')
	ax.yaxis.set_ticks_position('none')
	ax.set_facecolor('xkcd:black')
	plt.subplot(212)
	plt.plot(ta[0:frame],ke[0:frame],'xkcd:red',lw=1.0)
	plt.plot(ta[0:frame],pe[0:frame],'xkcd:cerulean',lw=1.0)
	plt.plot(ta[0:frame],E[0:frame],'xkcd:bright green',lw=1.5)
	plt.xlim([0,tf])
	plt.title("Energy")
	ax=plt.gca()
	ax.legend(['T','V','E'],labelcolor='w',frameon=False)
	ax.set_facecolor('xkcd:black')

ani=animation.FuncAnimation(fig,run,frames=nframes)
#writervideo = animation.FFMpegWriter(fps=nfps)
#ani.save('mass_spring_triplet.mp4', writer=writervideo)
plt.show()
















