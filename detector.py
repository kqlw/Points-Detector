import numpy as np
import cv2


class ImgSearch():
  def __init__(self) -> None:
    # параметры для гистограммы
    self.__bins = None
    self.__channels = None
    self.__ranges = None

    # маска
    self.__mask = None

    # параметры для фильтра
    self.__neighborhood_diameter = None
    self.__sigmaColor = None
    self.__sigmaSpace = None

  @property
  def historgrammParametrs(self):
    return [    self.__bins, self.__channels, self.__ranges]

  @historgrammParametrs.setter
  def historgrammParametrs(self, value):
    channels, ranges, bins = value
    self.__bins = bins
    self.__channels = channels
    self.__ranges = ranges

  @property
  def bins(self):
    return  self.__bins

  @bins.setter
  def bins(self, bins):
     self.__bins = bins


  @property
  def mask(self):
    return self.__mask

  @mask.setter
  def mask(self, mask):
    self.__mask = mask
    self.__maskSize = len(mask[ mask == 255 ])
    if self.__maskSize == 0:
      self.__maskSize = 1
      print("Нет выделенных элементов! Размер маски 1")


  @property
  def historgramm(self):
    return self.__historgramm
  @property
  def backprojection_map(self):
    return self.__backprojection_map

  @backprojection_map.setter
  def backprojection_map(self, map):
    self.__backprojection_map = map



  def takeSample(self, img):
    self.__historgramm = cv2.calcHist([img], self.__channels , self.__mask, self.__bins, self.__ranges)




  def search(self, img, scale = 1):
    # обратное проецирование гистограммы
    # ipdb.set_trace(context=10)
    self.__backprojection_map = cv2.calcBackProject([img],  self.__channels, self.__historgramm, self.__ranges, scale)


    # ipdb.set_trace(context=10)
    # вероятность того что пиксель принадлежит маске
    self.__backprojection_map =  self.__backprojection_map.astype(np.float32) / self.__maskSize


    threshold = 0.01

    indices = np.nonzero(self.__backprojection_map > threshold)

    x = indices[1].mean()
    y = indices[0].mean() 

    return [x, y]


  






