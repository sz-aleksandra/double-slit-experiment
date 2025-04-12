import sys
import numpy as np
import matplotlib.pyplot as plt
import imageio as iio

'''
execution examples: 
		1. python3 double_slit.py (default conditions - don't know what they should be; BE CAREFUL: IT CREATES AND SAVES 100 PLOT FILES AND 1 ANIMATION)
        2. python3 double_slit.py V0 T K Lx Ly Nx Ny p0 sigma_x sigma_y x1 x2 y1 y2 y3 y4 x0 y0 (own arguments/ conditions; BE CAREFUL: IT CREATES AND SAVES K PLOT FILES AND 1 ANIMATION)
        3. python3 double_slit.py 500 10 100 5 5 50 50 10 0.5 0.5 2.4 2.6 1.25 2.25 2.75 3.75 1.25 2.5 (example console input arguments; BE CAREFUL: IT CREATES AND SAVES 100 PLOT FILES AND 1 ANIMATION)
'''

'''calculates potential in (x,y)'''
def Vxy(x, y, Lx, Ly, x1, x2, y1, y2, y3, y4, V0):
    V = 0
    if x < 0 or x > Lx or y < 0 or y > Ly:
        V = float('inf')
    if (x1 <= x <= x2 and 0 <= y <= y1) or (x1 <= x <= x2 and y2 <= y <= y3) or (x1 <= x <= x2 and y4 <= y <= Ly):
        V = V0
    return V

'''calculates coefficient rx for G and H matrices'''
def rx(delta_t, delta_x):
    return -((1j * delta_t) / (2.0 * (delta_x ** 2)))

'''calculates coefficient ry for G and H matrices'''
def ry(delta_t, delta_y):
    return -((1j * delta_t) / (2.0 * (delta_y ** 2)))

'''calculates coefficient aij for G and H matrices'''
def aij(i, j, delta_t, delta_x, delta_y, Lx, Ly, x1, x2, y1, y2, y3, y4, V0):
    return (1 - 2 * rx(delta_t, delta_x) - 2 * ry(delta_t, delta_y) + 1j * delta_t * Vxy(i * delta_x, j * delta_y, Lx, Ly, x1, x2, y1, y2, y3, y4, V0) / 2.0)

'''calculates coefficient aij for G and H matrices'''
def bij(i, j, delta_t, delta_x, delta_y, Lx, Ly, x1, x2, y1, y2, y3, y4, V0):
    return (1 + 2 * rx(delta_t, delta_x) + 2 * ry(delta_t, delta_y) - 1j * delta_t * Vxy(i * delta_x, j * delta_y, Lx, Ly, x1, x2, y1, y2, y3, y4, V0) / 2.0)

'''calculates G matrix'''
def create_G(Nx, Ny, delta_t, delta_x, delta_y, Lx, Ly, x1, x2, y1, y2, y3, y4, V0):
    n = (Nx + 1) * (Ny + 1)
    G = np.zeros((n, n), dtype=np.complex128)
    for j in range(Ny + 1):
        for i in range(Nx + 1):
            l = j * (Nx + 1) + i
            if 0 < i < Nx and 0 < j < Ny:
                G[l, l] = aij(i, j, delta_t, delta_x, delta_y, Lx, Ly, x1, x2, y1, y2, y3, y4, V0)
                G[l, l + 1] = rx(delta_t, delta_x)
                G[l, l - 1] = rx(delta_t, delta_x)
                G[l, l + Nx + 1] = ry(delta_t, delta_y)
                G[l, l - Nx - 1] = ry(delta_t, delta_y)
            else:
                G[l, l] = 1
    return G

'''calculates H matrix'''
def create_H(Nx, Ny, delta_t, delta_x, delta_y, Lx, Ly, x1, x2, y1, y2, y3, y4, V0):
    n = (Nx + 1) * (Ny + 1)
    H = np.zeros((n, n), dtype=np.complex128)
    for j in range(Ny + 1):
        for i in range(Nx + 1):
            l = j * (Nx + 1) + i
            if 0 < i < Nx and 0 < j < Ny:
                H[l, l] = bij(i, j, delta_t, delta_x, delta_y, Lx, Ly, x1, x2, y1, y2, y3, y4, V0)
                H[l, l + 1] = -rx(delta_t, delta_x)
                H[l, l - 1] = -rx(delta_t, delta_x)
                H[l, l + Nx + 1] = -ry(delta_t, delta_y)
                H[l, l - Nx - 1] = -ry(delta_t, delta_y)
            else:
                H[l, l] = 1
    return H

'''calculates U matrix'''
def calculate_U(G, H):
    return np.dot(np.linalg.inv(G), H)

'''creates initial wave function'''
def psi0(Nx, Ny, sigma_x, sigma_y, delta_x, x0, delta_y, y0, p0):
    psi = np.zeros((Nx + 1) * (Ny + 1), dtype=np.complex128)
    for j in range(Ny + 1):
        for i in range(Nx + 1):
            expression_1 = 1 / np.sqrt(np.pi * sigma_x * sigma_y)
            exponent_1 = -(1 / (2 * (sigma_x ** 2)) * ((i * delta_x - x0) ** 2)) - (1 / (2 * (sigma_y ** 2)) * ((j * delta_y - y0) ** 2))
            exponent_2 = 1j * p0 * (i * delta_x - x0)
            psi[j * (Nx + 1) + i] = expression_1 * np.exp(exponent_1) * np.exp(exponent_2)
            
    normalization_factor = np.linalg.norm(psi)
    psi /= normalization_factor
    return psi

'''calculates wave function in next step of time evolution'''
def calculate_psik_plus_one(U, psik):
    return np.dot(U, psik)

'''calculates probability density of wave function'''
def calculate_Probability_density(psik, Nx, Ny):
    psik = np.abs(psik) ** 2
    P = psik.reshape((Nx + 1, Ny + 1))
    return P


'''plots probability density of wave function and barrier with two slits'''
def plot_P(P, k, Lx, Nx, Ly, Ny, x1, x2, y1, y2, y3, y4, delta_t):
    x = np.linspace(0, Lx, Nx + 1)
    y = np.linspace(0, Ly, Ny + 1)
    X, Y = np.meshgrid(x, y)

    plt.figure()
    plt.contourf(X, Y, P, cmap='cool') #or plt.pcolormesh(X, Y, P, cmap='cool')
    plt.colorbar(label='Probability Density')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title(f'Probability Density Function t = {np.round(k * delta_t, 2)}')
    
    slit_vertices1 = [(x1, y1), (x1, 0), (x2, 0), (x2, y1)]
    slit_vertices2 = [(x1, y2), (x1, y3), (x2, y3), (x2, y2)]
    slit_vertices3 = [(x1, Ly), (x1, y4), (x2, y4), (x2, Ly)]
    
    plt.fill([vertex[0] for vertex in slit_vertices1], [vertex[1] for vertex in slit_vertices1], color='black')
    plt.fill([vertex[0] for vertex in slit_vertices2], [vertex[1] for vertex in slit_vertices2], color='black')
    plt.fill([vertex[0] for vertex in slit_vertices3], [vertex[1] for vertex in slit_vertices3], color='black')

    filename = f'plot_{k}_t={np.round(k * delta_t, 2)}.png'
    plt.savefig(filename)
    plt.close()
    
    return filename

'''simulates double slit experiment, saves it as plots and a animation;
initializes wave function, creates G and H matrix according to given or default conditions,
uses them to calculate U matrix, used to find wave function over time evolution,
saves plots of next time steps, loads them to create and save an animation'''
def main():
    if len(sys.argv) == 19:
        V0, T, K, Lx, Ly, Nx, Ny, p0, sigma_x, sigma_y, x1, x2, y1, y2, y3, y4, x0, y0 = map(float, sys.argv[1:])
        K = int(K)
        Nx = int(Nx)
        Ny = int(Ny)
    else:
        V0 = 2000
        T, K = 10, 100
        Lx, Ly = 2, 2
        Nx, Ny = 50, 50
        p0 = 10
        sigma_x, sigma_y = Lx / 10.0, Ly / 10.0
        x1, x2 = 0.95 * Lx / 2.0, 1.05 * Lx / 2.0
        y1, y2, y3, y4 = 0.5 * Ly / 2.0, 0.85 * Ly / 2.0, 1.15 * Ly / 2.0, 1.5 * Ly / 2.0
        x0, y0, t0 = Lx / 4.0, Ly / 2.0, 0
        
    delta_t = T / K
    delta_x = Lx / Nx
    delta_y = Ly / Ny

    psi = psi0(Nx, Ny, sigma_x, sigma_y, delta_x, x0, delta_y, y0, p0)
    
    G = create_G(Nx, Ny, delta_t, delta_x, delta_y, Lx, Ly, x1, x2, y1, y2, y3, y4, V0)
    H = create_H(Nx, Ny, delta_t, delta_x, delta_y, Lx, Ly, x1, x2, y1, y2, y3, y4, V0)

    U = calculate_U(G, H)
    
    image_files = []
    for k in range(K):
        P = calculate_Probability_density(psi, Nx, Ny)
        image_files.append(plot_P(P, k, Lx, Nx, Ly, Ny, x1, x2, y1, y2, y3, y4, delta_t))
        psi = calculate_psik_plus_one(U, psi)

    images = []
    for filename in image_files:
        images.append(iio.v3.imread(filename))
    iio.mimsave('double_slit_probability_density.gif', images, fps=5)

if __name__ == "__main__":
	main()
