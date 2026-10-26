import numpy as np

class Interpolator():
    def __init__(self):
        pass

    # проверка на тип кривой: 1 - спираль, 0 - кривая
    def isSpiral(self, x, y, threshold=0.8):
        # 1. Центр кривой — средние координаты
        xc, yc = np.mean(x), np.mean(y)
        
        # 2. Переводим в полярные координаты
        x_rel = x - xc
        y_rel = y - yc
        r = np.sqrt(x_rel**2 + y_rel**2)
        theta = np.arctan2(y_rel, x_rel)
        
        # 3. Убираем скачки угла через ±π
        theta_unwrapped = np.unwrap(theta)
        
        # 4. Монотонность угла
        dtheta = np.diff(theta_unwrapped)
        monotonic_ratio = np.sum(dtheta > 0) / len(dtheta)
        
        # 5. Корреляция радиуса и угла
        corr = np.corrcoef(r, theta_unwrapped)[0, 1]
        
        # 6. Признак спирали
        is_spiral = (abs(corr) > threshold) and (monotonic_ratio > 0.7 or monotonic_ratio < 0.3)
        return is_spiral

    # Кубический сплайн (натуральный)
    def cubicSpline(self, x, y, x_new):
        n = len(x)
        h = np.diff(x)
        h[h == 0] = 1e-9  # защита от деления на ноль

        # Система для коэффициентов
        alpha = np.zeros(n)
        for i in range(1, n-1):
            alpha[i] = 3 * ((y[i+1]-y[i])/h[i] - (y[i]-y[i-1])/h[i-1])
        
        l = np.ones(n)
        mu = np.zeros(n)
        z = np.zeros(n)
        for i in range(1, n-1):
            l[i] = 2*(x[i+1]-x[i-1]) - h[i-1]*mu[i-1]
            mu[i] = h[i]/l[i]
            z[i] = (alpha[i] - h[i-1]*z[i-1])/l[i]
        
        c = np.zeros(n)
        b = np.zeros(n-1)
        d = np.zeros(n-1)
        a = y[:-1]
        for j in reversed(range(n-1)):
            c[j] = z[j] - mu[j]*c[j+1]
            b[j] = (y[j+1]-y[j])/h[j] - h[j]*(c[j+1]+2*c[j])/3
            d[j] = (c[j+1]-c[j])/(3*h[j])
        
        # Интерполяция
        y_new = np.zeros_like(x_new)
        for i, xi in enumerate(x_new):
            j = np.searchsorted(x, xi) - 1
            if j < 0:
                j = 0
            elif j >= n-1:
                j = n-2
            dx = xi - x[j]
            y_new[i] = a[j] + b[j]*dx + c[j]*dx**2 + d[j]*dx**3
        
        return y_new

    # Основной метод интерполяции
    def interpolate(self, x, y, x_new):
        x = np.asarray(x)
        y = np.asarray(y)

        # если спираль — интерполяция по длине дуги
        if self.isSpiral(x, y):
            # параметр длины дуги
            ds = np.sqrt(np.diff(x)**2 + np.diff(y)**2)
            s = np.hstack(([0], np.cumsum(ds)))
            s_new = np.linspace(0, s[-1], 200)
            
            x_smooth = self.cubicSpline(s, x, s_new)
            y_smooth = self.cubicSpline(s, y, s_new)

            # проверка направления вращения
            xc, yc = np.mean(x), np.mean(y)
            theta_orig = np.unwrap(np.arctan2(y - yc, x - xc))
            theta_interp = np.unwrap(np.arctan2(y_smooth - np.mean(y_smooth),
                                                x_smooth - np.mean(x_smooth)))
            
            dir_orig = np.sign(np.mean(np.diff(theta_orig)))
            dir_interp = np.sign(np.mean(np.diff(theta_interp)))
            
            # если направление изменилось — инвертируем
            if dir_orig != 0 and dir_interp != 0 and dir_orig != dir_interp:
                x_smooth = x_smooth[::-1]
                y_smooth = y_smooth[::-1]

            return np.column_stack((x_smooth, y_smooth))

        else:
            # обычная кривая — по x
            order = np.argsort(x)
            x_sorted, y_sorted = x[order], y[order]
            y_new = self.cubicSpline(x_sorted, y_sorted, x_new)
            return np.column_stack((x_new, y_new))
