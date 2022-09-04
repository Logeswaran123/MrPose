import math

class Operation():
    def __init__(self) -> None:
        pass

    def angle_of_singleline(self, point1, point2):
        """ Calculate angle of a single line """
        xDiff = point2[0] - point1[0]
        yDiff = point2[1] - point1[1]
        return math.degrees(math.atan2(yDiff, xDiff))
    
    def angle(self, point1, point2, point3):
        """ Calculate angle between two lines """
        if(point1==(0,0) or point2==(0,0) or point3==(0,0)):
            return 0
        numerator = point2[1] * (point1[0] - point3[0]) + point1[1] * (point3[0] - point2[0]) + point3[1] * (point2[0] - point1[0])
        denominator = (point2[0] - point1[0]) * (point1[0] - point3[0]) + (point2[1] - point1[1]) * (point1[1] - point3[1])
        try:
            ang = math.atan(numerator/denominator)
            ang = ang * 180 / math.pi
            if ang < 0:
                ang = 180 + ang
            return ang
        except:
            return 90.0