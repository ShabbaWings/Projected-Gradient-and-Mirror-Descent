import numpy as np
from sympy import diff, Symbol, Matrix
import matplotlib
import matplotlib.pyplot as plt

# матрицы коэффициентов
def gen_biases(n=3):
    A = np.array([[1, 0, 1], [0, 1, 0], [1, 0, 1]])
    b = np.array([1, 0, 1]).reshape((1, 3))
    return A, b


# возвращает значение функции в точке
def func_eval(x, A, b):
    x = x.reshape((1, n))
    return (x @ A @ x.T + b @ x.T)[0, 0]


# получает градиент в виде формулы
def get_gradient(A, b, n=3):
    x = Matrix([Symbol('x[' + str(i) + ']') for i in range(n)])
    f = x.T * A * x + b * x
    return [diff(f, x[i])[0, 0] for i in range(n)]


# подставляет численные координаты вектора вместо переменных вида 'x[i]'
# в формулу градиента и возвращает строку вида 2*1+2*2+3*1
def replacer(x, s):
    for i in range(len(x)):
        s = s.replace('x[' + str(i) + ']', str(x[i]))
    return s


# считает значение градиента в точке
def eval_gradient(x_0, A, b, n=3):
    gradient = get_gradient(A, b)
    return np.array([eval(replacer(x_0, str(gradient[i]))) for i in range(n)])


# проекция на симплекс
def projection(v, n=3):
    if v.sum() == 1 and np.alltrue(v >= 0):
        return v
    u = np.sort(v)[::-1]
    cssv = np.cumsum(u)
    rho = np.nonzero(u * np.arange(1, n+1) > (cssv - 1))[0][-1]
    theta = (cssv[rho] - 1) / (rho + 1.0)
    w = (v - theta).clip(min=0)
    return w


# градиентный спуск
def gradientDescent(x_0, lamda, A, b, eps=1e-30, maxStep=1000):
    points = []
    gradients = []
    points.append(x_0)
    gradients.append(np.dot(np.dot(x_0.T, A), x_0) + np.dot(b, x_0)[0])
    x = np.copy(x_0)
    step = 0
    while (step == 0 or (abs(func_eval(points[-1], A, b) - func_eval(points[-2], A, b)) >= eps) and step <= maxStep):
        step += 1
        g = eval_gradient(x, A, b)
        x -= lamda * g
        if (np.sum(x) != 1 or len([i for i in x if i<0])):
            x = projection(x)
        points.append(x)
        gradients.append(np.dot(np.dot(x.T, A), x) + np.dot(b, x)[0])
    return x, np.asarray(points), np.asarray(gradients)


# зеркальный спуск
def mirrorDescent(x_0, lamda, A, b, eps=1e-30, maxStep=1000):
    points = []
    gradients = []
    points.append(x_0)
    x = np.copy(x_0)
    gradients.append(np.dot(np.dot(x_0.T, A), x_0) + np.dot(b, x_0)[0])
    step = 0
    while (step == 0 or (abs(func_eval(points[-1], A, b) - func_eval(points[-2], A, b)) >= eps) and step <= maxStep):
        step += 1
        g = eval_gradient(x, A, b)
        y = x * np.exp(-lamda * g)
        x = y / np.linalg.norm((y), ord=1)
        points.append(x)
        gradients.append(np.dot(np.dot(x.T, A), x) + np.dot(b, x)[0])
    return x, np.asarray(points), np.asarray(gradients)


n = 3
x_0 = np.array([0.5, 0.5, 0.5], dtype=np.float64)
A, b = gen_biases()
lamda = 0.00001
for i in range (5):
    min_pmt, points, gradients = gradientDescent(x_0, lamda, A=A, b=b)
    min_pmt_1, points_1, gradients_1 = mirrorDescent(x_0, lamda, A=A, b=b)
    gradients = np.log(gradients)
    gradients_1 = np.log(gradients_1)
    strr = "PGD lamda=" + str(lamda)
    strr_1 = "MD lamda=" + str(lamda)
    plt.plot(gradients, label=strr)
    plt.plot(gradients_1, label=strr_1)
    lamda *= 10
plt.xlabel('iterations')
plt.ylabel('accuracy')
plt.grid()
plt.legend()
plt.show()