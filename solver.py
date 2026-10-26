import numpy as np
import cv2
import matplotlib.pyplot as plt
from interpolation import Interpolator
from detector import ImgSearch

class Solver(object):
    def __init__(self):
        self.interpolator = Interpolator()
        
        # гистограммы для каждой серии
        self.seriesHistogram = []
        self.calculateSeriesHist()
        # интреполяция для классов

        # загрузка образцов
        self.samplesIMG = [i for i in range(0, 9)]
        self.samplesMask = [i for i in range(0, 9)]

        self.load_sample()

        # текущая серия
        self.currentSeries = None
        
        self.imgSreach = ImgSearch()


        # поиск, должен вернуть записать (x, y) точки в points_info
        self.alg = self.find_points_coord
        self.points_info = []  # internal var (for example)


    def calculateSeriesHist(self):
      
      for s in range(1, 10):
        
        for i in range(1, 31):
          img_path = f"data/000{s}/opencv_images/opencv_image_{i}.png"
          img = cv2.imread(img_path)
          
          # 3D-гистограмма для BGR изображения
          hist = cv2.calcHist([img], [0,1,2], mask=None, histSize=[32,32,32], ranges=[0,256,0,256,0,256])
          
          # нормализация 
          cv2.normalize(hist, hist)
          
          if i == 1:
              hist_total = hist
          else:
              hist_total += hist 

        self.seriesHistogram.append(hist_total)
        hist_total = 0
        

    def getSeriesHistogramIndex(self, img):
        res = []
        # 3D-гистограмма для текущего изображения
        hist = cv2.calcHist([img], [0,1,2], mask=None, histSize=[32,32,32], ranges=[0,256,0,256,0,256])

        for sh in range(0, 9):
          score_corr = cv2.compareHist(hist, self.seriesHistogram[sh], cv2.HISTCMP_INTERSECT)
          res.append(score_corr)
        
        return res.index( max(res) )

    # загрузить фото для взятия образцов
    def load_sample(self):
      
      # образец для 1ой серии
      # маска для 1ой серии HSV
      # mask[345:355, 10:20] = 255

      url_1 = "data/0001/opencv_images/opencv_image_2.png"
      img_1 =  cv2.imread(url_1)
      img_1  = cv2.cvtColor(img_1, cv2.COLOR_BGR2HSV)
      
      self.samplesIMG[0] = img_1      
      
      self.samplesMask[0] = np.zeros(img_1.shape[:2], dtype=np.uint8)
      self.samplesMask[0][345:355, 10:20] = 255
      


      # # 2я серия
      # mask[245:255, 313:322] = 255

      url_2 = "data/0002/opencv_images/opencv_image_1.png"
      img_2 =  cv2.imread(url_2)
      img_2 = cv2.cvtColor(img_2, cv2.COLOR_BGR2HSV)
      
      self.samplesIMG[1] = img_2        
      
      self.samplesMask[1] = np.zeros(img_2.shape[:2], dtype=np.uint8)
      self.samplesMask[1][245:255, 313:322] = 255
      

      # # 3я серия
      # mask[ 233:247, 287:300 ] = 255

      url_3 = f"data/0003/opencv_images/opencv_image_18.png"
      img_3 =  cv2.imread(url_3)
      img_3 = cv2.cvtColor(img_3, cv2.COLOR_BGR2HSV)
      
      self.samplesIMG[2] = img_3        
      
      self.samplesMask[2] = np.zeros(img_3.shape[:2], dtype=np.uint8)
      self.samplesMask[2][ 233:247, 287:300] = 255

      
      # 4я серия 

      # маска для 4ой серии HSV
      # mask[ 272:282, 133:150 ] = 255

      url_4 = f"data/0004/opencv_images/opencv_image_10.png"
      img_4 =  cv2.imread(url_4)
      img_4 = cv2.cvtColor(img_4, cv2.COLOR_BGR2HSV)
      
      self.samplesIMG[3] = img_4        
      
      self.samplesMask[3] = np.zeros(img_4.shape[:2], dtype=np.uint8)
      self.samplesMask[3][ 272:282, 133:150 ] = 255


      # 5я серия ПРОБЛЕМА 
      url_5 = f"data/0005/opencv_images/opencv_image_27.png"
      img_5 =  cv2.imread(url_5)
      # img_5 = cv2.cvtColor(img_4, cv2.COLOR_BGR2HSV)
      
      self.samplesIMG[4] = img_5        
      
      self.samplesMask[4] = np.zeros(img_5.shape[:2], dtype=np.uint8)
      self.samplesMask[4][   250:270, 425:435 ] = 255



      # 6я серия ПРОБЛЕМА
      # маска 6ая ФОРМАТ BGR
      # mask[ 262:277, 165:200 ] = 255

      url_6 = f"data/0006/opencv_images/opencv_image_10.png"
      img_6 =  cv2.imread(url_6)
      img_6 = cv2.cvtColor(img_6, cv2.COLOR_BGR2HSV)

      self.samplesIMG[5] = img_6        
      
      self.samplesMask[5] = np.zeros(img_6.shape[:2], dtype=np.uint8)
      self.samplesMask[5][ 250:290, 180:190] = 255


      # 7я серия 
      # 7 маска GRAY
      # mask[220:230, 55:74] = 255

      url_7 = f"data/0007/opencv_images/opencv_image_5.png"
      img_7 =  cv2.imread(url_7)
      img_7 = cv2.cvtColor(img_7, cv2.COLOR_BGR2GRAY)

      self.samplesIMG[6] = img_7        
      
      self.samplesMask[6] = np.zeros(img_7.shape[:2], dtype=np.uint8)
      self.samplesMask[6][ 220:230, 55:74 ] = 255

      # 8я серия  ПРОБЛЕМА
      url_8 = f"data/0008/opencv_images/opencv_image_29.png"
      img_8 =  cv2.imread(url_8)

      clahe = cv2.createCLAHE(clipLimit=0, tileGridSize=(16,16))

      img_8 = cv2.bilateralFilter(img_8, d=15, sigmaColor=100, sigmaSpace=100)  

      l, a, b = cv2.split(img_8)
      l_clahe = clahe.apply(l)
      img_8 = cv2.merge((l_clahe, a, b))

      self.samplesIMG[7] = img_8        
      
      self.samplesMask[7] = np.zeros(img_8.shape[:2], dtype=np.uint8)
      self.samplesMask[7][ 175:225, 150:174 ] = 255


      # 9я серия ПРОБЛЕМА 
      url_9 = f"data/0009/opencv_images/opencv_image_29.png"
      img_9 =  cv2.imread(url_9)
      img_9 = cv2.cvtColor(img_9, cv2.COLOR_BGR2GRAY)

      self.samplesIMG[8] = img_9        
      
      self.samplesMask[8] = np.zeros(img_9.shape[:2], dtype=np.uint8)
      self.samplesMask[8][ 220:230, 55:74 ] = 255

    # ПОИСК ТОЧКИ
    def find_points_coord(self, img):

      # определение текущей серии
      if self.currentSeries == None:
        self.currentSeries = self.getSeriesHistogramIndex(img)
        
        match self.currentSeries:
          # HSV
          case 0:
            self.imgSreach.historgrammParametrs = ([0, 1], [0, 180, 0, 256], [180, 256])
          
          # HSV
          case 1:
            self.imgSreach.historgrammParametrs = ([0, 1], [0, 180, 0, 256], [180, 256])
          
          # HSV
          case 2:
            self.imgSreach.historgrammParametrs = ([0, 1], [0, 180, 0, 256], [180, 256])
            
          case 3:
            self.imgSreach.historgrammParametrs = ([0, 1], [0, 180, 0, 256], [180, 256])

          case 4:
            self.imgSreach.historgrammParametrs = ([0, 1, 2], [0, 256, 0, 256, 0, 256], [256, 256, 256])
            
          case 5:
            self.imgSreach.historgrammParametrs = ([0, 1], [0, 180, 0, 256], [180, 256])

          # параметры GRAY
          case 6:
            self.imgSreach.historgrammParametrs = ([0], [0, 256], [256])
            
          case 7:
            # img = cv2.bilateralFilter(img, d=15, sigmaColor=70, sigmaSpace=50)
            self.imgSreach.historgrammParametrs = ([0, 1, 2], [0, 256, 0, 256, 0, 256], [256, 256, 256])

          case 8:
            self.imgSreach.historgrammParametrs = ([0], [0, 256], [256])

        # делаем слепок
        self.imgSreach.mask = self.samplesMask[ self.currentSeries ]
        self.imgSreach.takeSample( self.samplesIMG[ self.currentSeries ] )

      # обработка фото в зависимости от серии
      match self.currentSeries:
        case 0:
          img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
          
        case 1:
          img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

        case 2:
          img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
          
        case 3:
          img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

        case 4:
          pass
          
        case 5:
          img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

        case 6:
          pass
          
        case 7:
          clahe = cv2.createCLAHE(clipLimit=0, tileGridSize=(16,16))

          img = cv2.bilateralFilter(img, d=15, sigmaColor=100, sigmaSpace=100)  

          l, a, b = cv2.split(img)
          l_clahe = clahe.apply(l)
          img = cv2.merge((l_clahe, a, b))

        case 8:
          pass

      coord = self.imgSreach.search(img)

      self.points_info.append(coord)
    


    def solve(self, image_bgr: np.ndarray):
        result = self.alg(image_bgr)

    # ИНТЕРПОЛЯЦИЯ ТРАЕКТОРИИ
    def get_trajectory(self):

        pred_points = np.array(self.points_info)
        
        train_data = np.array(self.points_info)
        
        x = train_data[:, 0]
        y = train_data[:, 1]
        
        x_new  = np.linspace(x.min(), x.max(), 100)
        pred_traj = self.interpolator.interpolate(x, y, x_new)
        
        self.currentSeries = None
        self.points_info = []
        return pred_points, pred_traj