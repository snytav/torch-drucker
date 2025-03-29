import numpy as np
def Poisson1D( v, L ):
  # Solve 1-d Poisson equation:
  #      d^u / dx^2 = v   for  0 <= x <= L
 # # using spectral method
  J = v.shape[0]
  #  Fourier transform source term
  v_ml = np.loadtxt('v.txt')
  eps_v = np.max(np.abs(v_ml - v))
  v_tilde = np.fft.fft(v)
  vm = np.loadtxt('v_real.txt')
  eps_v_tilde = np.max(np.abs(vm-np.real(v_tilde)))
  # vector of wave numbers
  k1 = np.linspace(0, int(J / 2) - 1, int(J / 2))
  k2 = np.linspace(-int(J / 2), -1, int(J / 2))
  k = (2*np.pi/L)*np.concatenate((k1,k2))
  # k = np.zeros(J)
#  k = (2*np.pi/L)*[0:(J/2-1),(-J/2):(-1)]
  k[0]= 1
  # Calculate Fourier transform of u
  u_tilde = -v_tilde/np.power(k,2.0)
  # Inverse Fourier transform to obtain u
  u = np.real(np.fft.ifft(u_tilde));
  # Specify arbitrary constant by forcing corner u = 0;
  u = u - u[0]
  return u
