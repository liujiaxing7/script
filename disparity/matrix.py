import numpy as np

R2=np.array([[9.9867037336427134e-01, -1.2400746793515377e-02,
       -5.0037054703791523e-02], [1.2265183467690229e-02,
       9.9992023310495382e-01, -3.0154107247335749e-03],
       [5.0070456748176444e-02, 2.3976876981914906e-03,
       9.9874280996397224e-01]])

R1=np.array([[9.9809779359517481e-01, -2.1767895829825780e-02,
       -5.7679746285726120e-02], [2.1924055279546953e-02,
       9.9975748391875963e-01, 2.0758488698801748e-03],
       [5.7620571157731990e-02, -3.3364741229429027e-03,
       9.9833297938106991e-01]])

P2=np.array([ [2.8380752815977678e+02, 0., 3.4049471571846959e+02,
       -1.4006794021370021e+01], [0., 2.8380752815977678e+02,
       2.1024559662966092e+02, 0.],[0., 0., 1., 0.]])

R=R2.T * R1
print("R:",R)

baseline=-P2[0,3]/P2[0,0]
print("baseline:",baseline)

t=R2.T * np.array([-baseline,1e-10,1e-10]).reshape([3,1])
print("t:",t)