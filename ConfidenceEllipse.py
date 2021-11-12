from matplotlib.patches import Ellipse
from scipy.stats import chi2
import numpy as np 
import math

# https://www.haya-programming.com/entry/2018/02/14/235500
# より
class ConfidenceEllipse:
    def __init__(self, m,cov, p=0.95):
       
        self.p = p

        self.means = m
        self.cov = cov

        su = (cov[0,0]+cov[1,1]+math.sqrt((cov[0,0]-cov[1,1])**2+4*cov[1,0]**2))/2
        sv = (cov[0,0]+cov[1,1]-math.sqrt((cov[0,0]-cov[1,1])**2+4*cov[1,0]**2))/2

        c = np.sqrt(chi2.ppf(self.p, 2))
        self.w= 2*c*math.sqrt(su)
        self.h = 2*c * np.sqrt(sv)
        self.theta = np.degrees(np.arctan(
            ((su - cov[0,0])/self.cov[0,1])))
        
    def get_params(self):
        return self.means, self.w, self.h, self.theta

    def get_patch(self, line_color="black", face_color="none", alpha=0):
        el = Ellipse(xy=self.means,
                     width=self.w, height=self.h,
                     angle=self.theta, color=line_color, alpha=alpha)
        el.set_facecolor(face_color)
        return el