from solver import Solver
import matplotlib.pyplot as plt
import numpy as np
import cv2

solver = Solver()

def check(s):
    for i in range(1, 31):
    
        img_path = f"data/000{s}/opencv_images/opencv_image_{i}.png"
        img =  cv2.imread(img_path)
        

        # img = cv2.cvtColor(img, cv2.COLOR_BGR2)
        solver.solve(img)
    
    ds = solver.currentSeries + 1
    fp, pp = solver.get_trajectory()

    plt.scatter(fp[:, 0], fp[:, 1])
    plt.plot(pp[:, 0], pp[:, 1])
    plt.title(f"Серия :{s}")
    plt.show()


for i in range(1, 10):
    check(i)

# Успешно обнаруживаются точки в серииях: 1, 2, 3, 4, 6, 7 


