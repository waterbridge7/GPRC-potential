import numpy as np
import pdb, scipy, time
from matplotlib import pyplot as plt
from matplotlib.colors import LogNorm
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel
from scipy.linalg import cholesky, cho_solve, solve_triangular

def comp_K(X, Y, params, tag):
  alpha, length_scale, sigma2 = params
  # X and Y must be a 2-dimensional array
  RBF_yy = RBF(length_scale, (1e-2, 1e3))
  # pdb.set_trace()
  K_ff = RBF_yy(X, Y)
  dist = np.tile(X,(1,Y.shape[0])) - np.tile(np.transpose(Y),(X.shape[0],1))
  if tag == "y_y":
    K = alpha * K_ff + 0.0077*np.eye(K_ff.shape[0])
  elif tag == "y_g":
    K = (dist)/length_scale**2* alpha * K_ff
  elif tag == "y_f":
    K = alpha* K_ff
  elif tag == "y_h":
    K = (dist**2/length_scale**4 - 1/length_scale**2)* alpha * K_ff
  elif tag == "f_y":
    K = alpha* K_ff
  elif tag == "f_f":
    K = alpha* K_ff 
  elif tag == "f_g":
    K = (dist)/length_scale**2* alpha * K_ff
  elif tag == "f_h":
    K = (dist**2/length_scale**4 - 1/length_scale**2)* alpha * K_ff
  elif tag == "g_y":
    K = -(dist)/length_scale**2* alpha * K_ff
  elif tag == "g_f":
    K = -(dist)/length_scale**2* alpha * K_ff
  elif tag == "g_g":
    K = (-dist**2/length_scale**4 + 1/(length_scale**2))*alpha*K_ff 
  elif tag == "g_h":
    K = (-dist**3/length_scale**6 + 3*dist/length_scale**4)*alpha*K_ff
  elif tag == "h_y":
    K = (dist**2/length_scale**4 - 1/length_scale**2)* alpha * K_ff
  elif tag == "h_f":
    K = (dist**2/length_scale**4 - 1/length_scale**2)* alpha * K_ff
  elif tag == "h_h":
    K = (3/length_scale**4 - 6*dist**2/length_scale**4 + dist**4/length_scale**8)*alpha*K_ff
  elif tag == "h_g":
    K = (dist**3/length_scale**6 - 3 * dist/length_scale**4)*alpha*K_ff
  else :
    print("error!")
  return K

def Com_SigR(x, length_scale, tag):
  length_scale = .1
  if tag == 'f':
    return np.diag(np.exp(x[:,0]**1/length_scale**2)-0.99999)
  elif tag=='g':
    return np.diag(np.exp(x[:,0]**1/length_scale**2)-0.99999)
  elif tag=='h':
    return np.diag(np.exp(x[:,0]**1/length_scale**2)-0.99999)
  else:
    print ("wrong tag")
    return -1
def K_rr(x,params,b,c):
    # pdb.set_trace()
    # K = comp_K(X_dataAndx_pre, X_dataAndx_pre,params, "h_h")\
    # + b**2* comp_K(X_dataAndx_pre, X_dataAndx_pre,params, "g_g")\
    # + c**2* comp_K(X_dataAndx_pre, X_dataAndx_pre,params,"f_f")\
    # + 2*b* comp_K(X_dataAndx_pre, X_dataAndx_pre,params,"h_g")\
    # + 2*c* comp_K(X_dataAndx_pre, X_dataAndx_pre,params,"h_f")\
    # + 2*b*c* comp_K(X_dataAndx_pre, X_dataAndx_pre,params,"g_f")\
    # + 1e-2*np.eye(X_dataAndx_pre.shape[0])
    K = 1**2*comp_K(x, x,params, "h_h")\
    + b**2* comp_K(x, x,params, "g_g")\
    + c**2* comp_K(x, x,params,"f_f")\
    + b* comp_K(x, x,params,"h_g")\
    + b* comp_K(x, x,params,"g_h")\
    + c* comp_K(x, x,params,"h_f")\
    + c* comp_K(x, x,params,"f_h")\
    + b*c* comp_K(x, x,params,"f_g")\
    + b*c* comp_K(x, x,params,"g_f")\
    + .0077*np.eye(x.shape[0])
#    + .0077*np.eye(x.shape[0])
#    + params[-1]*np.eye(x.shape[0])
    return K
def predict_derivative(x_pre, X_data, obs, params,bc, tag="first"):
  # Data_yAndr include y and r values of the  trainning data
    alpha, length_scale, sigma2 = params
    # add Initial conditions
    # IC = (np.array([[0]]), np.pi - 0.1, 0, 0.2-2*np.pi)
    IC = (np.array([[0]]), np.pi - 0.1, 0, 0.3-3*np.pi)
    x0, f0, g0, h0 = IC 
    Cov_YY = comp_K(X_data,X_data,params,"y_y")
    # pdb.set_trace()
    length = 3
    freq   = .1
    X_dataAndx_pre = np.row_stack((np.arange(max(0,x_pre[0,0]-length), min(10, x_pre[0,0]+length),freq)[:,np.newaxis], x_pre))
#    X_dataAndx_pre = np.arange(max(0,x_pre[0,0]-length), min(10, x_pre[0,0]+length),freq)[:,np.newaxis]
#    X_dataAndx_pre = X_data
#    X_dataAndx_pre = x_pre
    Data_yAndr = obs
    Data_yAndr = np.append(Data_yAndr,np.zeros((1,X_dataAndx_pre.shape[0])))
    # pdb.set_trace()

    b, c = bc
    Cov_Yr = comp_K(X_data, X_dataAndx_pre,params, "y_h")\
    + b**1* comp_K(X_data, X_dataAndx_pre,params, "y_g")\
    + c**1* comp_K(X_data, X_dataAndx_pre,params, "y_f")
    # Cov_Yr = c**1* comp_K(X_data, X_dataAndx_pre,params, "y_f")
    # pdb.set_trace()
    Cov_rr = K_rr(X_dataAndx_pre,params,b,c) 

    Cov_rY = Cov_Yr.T
    K_DD = np.column_stack((np.row_stack((Cov_YY, Cov_rY)), np.row_stack((Cov_Yr, Cov_rr))))    
    inv_K_DD = np.linalg.inv(K_DD)
#    L = cholesky(K_DD, lower = True)
    
#     np.einsum("ik,ik->k", y, cho_solve((L, True), y))
    if tag =="f":
      Cov_fh = comp_K(x_pre,X_dataAndx_pre,params,"f_h")
      Cov_fg = comp_K(x_pre,X_dataAndx_pre,params, "f_g")
      Cov_ffdata = comp_K(x_pre,X_dataAndx_pre,params,"f_f")
      Cov_fr = Cov_fh + b* Cov_fg + c**1* Cov_ffdata
      # Cov_fr = c**1* Cov_ffdata
      Cov_ff = comp_K(x_pre, x_pre,params,"f_f")

      Cov_fY = comp_K(x_pre,X_data,params,"f_y")
      Cov_fD = np.column_stack((Cov_fY, Cov_fr))
      mean = np.dot(np.dot(Cov_fD,inv_K_DD),Data_yAndr)
#      mean = np.dot( Cov_fD, cho_solve((L, True), Data_yAndr)[:,np.newaxis])[0]
      Cov  = Cov_ff - np.dot(np.dot(Cov_fD,inv_K_DD),np.transpose(Cov_fD))
      # pdb.set_trace()
      Cov_1 = np.linalg.inv(Cov)
      mean_r = f0
      Cov_r = Com_SigR(x_pre, length_scale, 'f')
      Cov_r_1 = np.linalg.inv(Cov_r)

      mc = np.linalg.inv((Cov_1 + Cov_r_1))*(Cov_1*mean + Cov_r_1*mean_r)
      Cov_c = np.linalg.inv((Cov_1 + Cov_r_1))

      # Cc=1/(np.linalg.det(2*np.pi*(Cov + Cov_r)))*np.exp(-0.5*(mean - mean_r)*np.linalg.inv(Cov + Cov_r)*(mean - mean_r))
      # pdb.set_trace()
    elif tag =="first":
      Cov_gh  = comp_K(x_pre,X_dataAndx_pre,params,"g_h")
      Cov_ggdata = comp_K(x_pre,X_dataAndx_pre,params,"g_g")
      Cov_gf = comp_K(x_pre,X_dataAndx_pre,params,"g_f")

      Cov_gr =Cov_gh + b**1* Cov_ggdata +c* Cov_gf
      # Cov_gr = b**1* Cov_ggdata 
      # pdb.set_trace()
      Cov_gg = comp_K(x_pre, x_pre,params, "g_g")

      Cov_gY = comp_K(x_pre,X_data,params,"g_y")
      Cov_gD = np.column_stack((Cov_gY, Cov_gr))

      # Cov_gD, K_DD, Data_yAndr = Cov_gY, Cov_YY, obs
      # pdb.set_trace()
      mean = np.dot(np.dot(Cov_gD,inv_K_DD),Data_yAndr)
#      mean = np.dot( Cov_gD, cho_solve((L, True), Data_yAndr)[:,np.newaxis])[0]
      Cov  = Cov_gg - np.dot(np.dot(Cov_gD,inv_K_DD),np.transpose(Cov_gD))
      
      Cov_1 = np.linalg.inv(Cov)
      mean_r = g0
      Cov_r = Com_SigR(x_pre, length_scale, 'g')
      Cov_r_1 = np.linalg.inv(Cov_r)

      mc = np.linalg.inv((Cov_1 + Cov_r_1))*(Cov_1*mean + Cov_r_1*mean_r)
      Cov_c = np.linalg.inv((Cov_1 + Cov_r_1))

    elif tag=="second":
      Cov_hhdata  = comp_K(x_pre,X_dataAndx_pre,params,"h_h")
      Cov_hg = comp_K(x_pre,X_dataAndx_pre,params,"h_g")
      Cov_hf = comp_K(x_pre,X_dataAndx_pre,params,"h_f")
      Cov_hr = Cov_hhdata + b*Cov_hg + c* Cov_hf
      # Cov_hr = Cov_hhdata
      Cov_hY = comp_K(x_pre,X_data,params,"h_y")
      Cov_hD = np.column_stack((Cov_hY, Cov_hr))
      Cov_hh = comp_K(x_pre, x_pre,params,"h_h")

      mean = np.dot(np.dot(Cov_hD,inv_K_DD),Data_yAndr)
#      mean = np.dot( Cov_hD, cho_solve((L, True), Data_yAndr)[:,np.newaxis])[0]
      Cov  = Cov_hh - np.dot(np.dot(Cov_hD,inv_K_DD),np.transpose(Cov_hD))
      Cov_1 = np.linalg.inv(Cov)
      mean_r = h0
      Cov_r = Com_SigR(x_pre, length_scale, 'h')
      Cov_r_1 = np.linalg.inv(Cov_r)

      mc = np.linalg.inv((Cov_1 + Cov_r_1))*(Cov_1*mean + Cov_r_1*mean_r)
      Cov_c = np.linalg.inv((Cov_1 + Cov_r_1))

#    return (mc, Cov_c)
    return (mean, Cov)

#def MLE_fun(params, x, obs, bc):
#  b, c = bc
#  Dim = obs.shape[0]
#  sigma2 = np.exp(params)[-1]
#  # MLE for hyper-paremeters
#  kernel_f =  comp_K(x,x,np.exp(params), "f_f")
#    # Compute log-likelihood (compare line 7)
#  kernel_r = K_rr(x,np.exp(params),b,c)
#  # try:
#  #   L = cholesky(kernel_f+sigma2*np.eye(Dim), lower = True)
#  # except ValueError:
#  #   pdb.set_trace()
#  L = cholesky(kernel_f+sigma2*np.eye(Dim), lower = True)
#  alpha = cho_solve((L, True), obs[:,np.newaxis])  # Line 3
#  log_likelihood_dims = -0.5 * np.einsum("ik,ik->k", obs[:,np.newaxis], alpha)
#  log_likelihood_dims -= np.log(np.diag(L)).sum()
#  # K = kernel_f+sigma2*np.eye(Dim)
#  # log_likelihood_dims = -0.5 * np.mat(obs[:,np.newaxis].T) * np.mat(np.linalg.inv(K))*np.mat(obs[:,np.newaxis])
#  # log_likelihood_dims = -np.log(np.diag(K)).sum()
#
#  log_likelihood_dims -= kernel_f.shape[0] / 2 * np.log(2 * np.pi)
#  log_likelihood = log_likelihood_dims.sum(-1)
#  
#  x_r =  x #np.arange(0,101,5)[:,np.newaxis]#x
#  kernel_r = K_rr(x_r,np.exp(params),b,c) +1e-10*np.eye(x_r.shape[0])
#  log_likelihood_dims_r  = - 0.5 * np.log(np.linalg.det(kernel_r)+1e-10)
#  log_likelihood_dims_r -= kernel_r.shape[0] / 2 * np.log(2 * np.pi)
#  log_likelihood_r       = log_likelihood_dims_r.sum(-1)
#
#  print('params:',np.exp(params),'Log',log_likelihood + log_likelihood_r,'log1:',log_likelihood,'log2:',log_likelihood_dims_r)
#  return -(log_likelihood + log_likelihood_r) 

def MLE_fun(params, x, obs,bc):
  # pdb.set_trace()
  params = np.exp(params)
#  print('params:',params)
  try:
      (b, c) = bc
      # MLE for hyper-paremeters
      kernel_f =  comp_K(x,x,params, "y_y")
        # Compute log-likelihood (compare line 7)
      # pdb.set_trace()
      # print (np.exp(params))
      L = cholesky(kernel_f + 1e-10*np.eye(x.shape[0]), lower = True)
      alpha = cho_solve((L, True), obs[:,np.newaxis])  # Line 3
      log_likelihood_dims = -0.5 * np.einsum("ik,ik->k", obs[:,np.newaxis], alpha)
      log_likelihood_dims -= np.log(np.diag(L)).sum()
      log_likelihood_dims -= kernel_f.shape[0] / 2 * np.log(2 * np.pi)
      log_likelihood = log_likelihood_dims.sum(-1)
      
      x_r =  x #np.arange(0,101,5)[:,np.newaxis]#x
      kernel_r = K_rr(x_r,params,b,c) +1e-10*np.eye(x_r.shape[0])
      L_r = cholesky(kernel_r, lower = True)
      log_likelihood_dims_r = -np.log(np.diag(L_r)).sum()
#      log_likelihood_dims_r  = - 0.5 * np.log(np.linalg.det(kernel_r)+1e-10)
#      log_likelihood_dims_r -= kernel_r.shape[0] / 2 * np.log(2 * np.pi)
    
#      print('Log',log_likelihood + log_likelihood_r,'log1:',log_likelihood,'log2:',log_likelihood_dims_r)
      return -(log_likelihood + log_likelihood_dims_r) 
  except:
#      print (' params error!')
      return -200
      


def Comp_RMS(x1,x2,x3,y1,y2,y3):
  print (np.sqrt((sum((x1-y1)**2) ) / x.shape[0]))
  print (np.sqrt((sum((x2-y2)**2) ) / x.shape[0]))
  print (np.sqrt((sum((x3-y3)**2) ) / x.shape[0]))

  return np.sqrt((sum((x1-y1)**2) +sum((x2-y2)**2)+sum((x3-y3)**2)) / (3*x.shape[0]))

# load and process data
def main(bc, x, obs, params_ops_my):
  b, c = bc
  hyper = 'GPRC'
  if hyper == 'GPRC':
      params0 = np.log(params_ops_my) # params0 = log(real_parameters)
      bounds = np.log([(.1,10),(.1,1),(1e-2, 1)])
      log_params_ops,value_ops,dd = scipy.optimize.fmin_l_bfgs_b(MLE_fun, params0, args=(x,obs, bc),bounds=bounds,\
      epsilon = 1e-3, approx_grad = 1)
      
      params_ops = np.exp(log_params_ops)
  elif hyper == 'GPR':
      params_ops = params_ops_my
#      params_ops = np.array([0.50, 1., 0.015])
  else:
      print ('something wrong')
  
  Cov_YY = comp_K(x,x,params_ops,"y_y")
  Cov_Yr = comp_K(x, x,params_ops, "y_h")\
  + b**1* comp_K(x, x,params_ops, "y_g")\
  + c**1* comp_K(x, x,params_ops, "y_f")
  Cov_rr = K_rr(x,params_ops,b,c) 
  Cov_rY = Cov_Yr.T

  K_DD = np.column_stack((np.row_stack((Cov_YY, Cov_rY)), np.row_stack((Cov_Yr, Cov_rr))))    
  inv_K_DD = np.linalg.inv(K_DD)  
  if np.linalg.cond(inv_K_DD)<5e3:
      mm_u, u_cov = np.array([]), np.array([])
      mm_g, g_cov = np.array([]), np.array([])
      mm_h, h_cov = np.array([]), np.array([])
      for xx in x:
        # pdb.set_trace()
        temp, temp_c = predict_derivative(xx[:,np.newaxis], x, obs, params_ops,bc,tag="f")
        mm_u = np.append(mm_u,temp)
        u_cov = np.append(u_cov,temp_c)
    
        temp, temp_c = predict_derivative(xx[:,np.newaxis], x, obs, params_ops,bc,tag="first")
        mm_g = np.append(mm_g,temp)
        g_cov = np.append(g_cov,temp_c)
    
        temp, temp_c = predict_derivative(xx[:,np.newaxis], x, obs, params_ops,bc,tag="second")
        mm_h = np.append(mm_h,temp)
        h_cov = np.append(h_cov,temp_c)
      
      l = np.exp(-1000* (np.mean((mm_h + b*mm_g + c*mm_u)**2) + np.mean((obs - mm_u)**2)))
      if l >1e-5:
          return 1e-20
      else:
          return l
       
  else:
      print(np.linalg.cond(inv_K_DD))
      return  np.exp(-1000*0.2)
#  if l < 1e-3:
#      l = 2
  # return l
#  pdb.set_trace()
  
  
def predict_result(bc, x, obs, params_ops_my,gpr):
  hyper = 'GPRC'
  if hyper == 'GPRC':
      params0 = np.log(params_ops_my) # params0 = log(real_parameters)
      bounds = np.log([(.1,10),(.1,1),(1e-2, 1)])
      log_params_ops,value_ops,dd = scipy.optimize.fmin_l_bfgs_b(MLE_fun, params0, args=(x,obs, bc),bounds=bounds,\
      approx_grad = 1)
      params_ops = np.exp(log_params_ops)
  elif hyper == 'GPR':
      params_ops = params_ops_my
#      params_ops = np.array([0.50, 1., 0.015])
  else:
      print ('something wrong')
  print (params_ops)
  mm_u, u_cov = np.array([]), np.array([])
  mm_g, g_cov = np.array([]), np.array([])
  mm_h, h_cov = np.array([]), np.array([])
  for xx in data[0,:][:,np.newaxis]:
    # pdb.set_trace()
    temp, temp_c = predict_derivative(xx[:,np.newaxis], x, obs, params_ops,bc,tag="f")
    mm_u = np.append(mm_u,temp)
    u_cov = np.append(u_cov,temp_c)

    temp, temp_c = predict_derivative(xx[:,np.newaxis], x, obs, params_ops,bc,tag="first")
    mm_g = np.append(mm_g,temp)
    g_cov = np.append(g_cov,temp_c)

    temp, temp_c = predict_derivative(xx[:,np.newaxis], x, obs, params_ops,bc,tag="second")
    mm_h = np.append(mm_h,temp)
    h_cov = np.append(h_cov,temp_c)
  
  save_derivatives = np.zeros((6,mm_u.shape[0]))
  save_derivatives[0,:] = mm_u
  save_derivatives[1,:] = u_cov
  save_derivatives[2,:] = mm_g
  save_derivatives[3,:] = g_cov
  save_derivatives[4,:] = mm_h
  save_derivatives[5,:] = h_cov
  
  gpr_solution = np.zeros((2,mm_u.shape[0]))
  (gpr_solution[0,:], gpr_solution[1,:]) = gpr.predict(data[0,:][:,np.newaxis],return_std='true')
  return (save_derivatives, gpr_solution)
  
#bc = (1, 3) Intel(R) Core(TM) i7-9700K CPU @ 3.60GHz
time_start = time.time()
data = np.load('data.npy')
index_data = np.arange(0,101,5)#np.random.randint(0, data.shape[1],100)
x_plot   = data[0, index_data, None]
obs_plot = data[1,index_data] 
x_obs , f_obs = x_plot[0:,:], obs_plot[0:]

np.random.seed(123) # for sigma=0.1
obs = f_obs + .1 * np.random.randn(f_obs.shape[0]) 
#np.random.seed(seed=None)
#np.random.seed(1) # for sigma=0.2
#obs = f_obs + .2 * np.random.randn(f_obs.shape[0]) 

kernel_u = 1.8 * RBF(length_scale=1.68, length_scale_bounds=(1e-1, 1e0)) \
    + WhiteKernel(noise_level=1e-2, noise_level_bounds=(1e-2, 1e+0))
gpr = GaussianProcessRegressor(kernel=kernel_u).fit(x_obs , obs)
params_ops_my = np.exp(gpr.kernel_.theta)

(save_derivatives, gpr_solution) = predict_result(np.array([1.,3.]), x_obs , obs, params_ops_my, gpr)
gpr_g = np.diff(gpr_solution[0,:])/0.1
gpr_h = np.diff(gpr_g)/0.1

#np.save('save_pre_gprc.npy',save_derivatives)
#np.save('gpr_solution.npy',gpr_solution)

#save_derivatives = np.load('save_pre_gprc.npy')
#gpr_solution = np.load('gpr_solution.npy')

# comparison between GPRC-potential and two-stage method
font2 = {'family' : 'Times New Roman',
'weight' : 'normal',
'size' : 10,}
plt.subplot(2,2,1)
plt.plot(data[0, :],data[1, :],'k')
plt.plot(data[0, :],save_derivatives[0,:],'r')
plt.plot(data[0, :], gpr_solution[0,:],'b')
plt.plot(x_obs, obs,'k*')
plt.xlabel(r'$ t $', font2)
plt.ylabel(r'$ u $', font2)

plt.subplot(2,2,2)
plt.plot(data[0, :],data[2, :],'k')
plt.plot(data[0, :],save_derivatives[2,:],'r')
plt.plot(data[0, :-1], gpr_g,'b')
plt.xlabel(r'$ t $', font2)
plt.ylabel(r'$ d u/d t $', font2)

plt.subplot(2,2,3)
plt.plot(data[0, :],data[3, :],'k')
plt.plot(data[0, 1:-1], gpr_h,'b')
plt.plot(data[0, :],save_derivatives[4,:],'r')
plt.xlabel(r'$ t $', font2)
plt.ylabel(r'$ {d^2 u/d t^2} $', font2)

plt.subplot(2,2,4)
plt.plot(data[0, :],data[3, :]+1*data[2, :]+3*data[1, :],'k')
plt.plot(data[0, :],save_derivatives[4,:] + 1*save_derivatives[2,:] + 3*save_derivatives[0,:],'r')
plt.plot(data[0, 1:-1],gpr_h + 1*gpr_g[0:-1] + 3*gpr_solution[0,1:-1],'b')
plt.xlabel(r'$ t $', font2)
plt.ylabel(r'$ r $', font2)

plt.subplots_adjust(wspace =0.3, hspace =0.35)
plt.savefig('examp1_0_1.png',dpi=300)

# MCMC for GPRC-potential
save_50mean = np.zeros((50,2))
for j in np.arange(0,1): 
    N = 5000
    x = np.zeros([N,2])
    x[0,:] = np.array([1.0,3.0])
    p_temp = main(x[0], x_obs , obs, params_ops_my)
    for i in np.arange(N-1):
        p_i = p_temp
        np.random.seed(seed=None)
        x_proposal = x[i,:] + .06 * np.random.randn(2)
        p_proposal = main(x_proposal, x_obs , obs, params_ops_my)
        A = min(1, p_proposal/p_i)
        np.random.seed(seed=None)
        if np.random.rand(1) <= A:
            x[i+1,:] = x_proposal
            p_temp = p_proposal
        else:
            x[i+1,:] = x[i,:]
            p_temp = p_i
        print('i:', i, 'x:',x[i+1,:], 'P:',p_temp)

    save_50mean[j,:]=np.mean(x,axis=0)

print('time cost:', time.time()-time_start)
np.save('bc_bignoise.npy',x)
plt.plot(x[:,0],x[:,1],'.')
plt.savefig('bc_bignoise.eps', dpi=300)
np.save('save_50mean_0_2.npy',save_50mean)


