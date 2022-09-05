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
    
    def dist_xy(self, point1, point2):
        """ Euclidean distance between two points point1, point2 """
        diff_point1 = (point1[0] - point2[0]) ** 2
        diff_point2 = (point1[1] - point2[1]) ** 2
        return (diff_point1 + diff_point2) ** 0.5

    def dist_x(self, point1, point2):
        return abs(point2[0] - point1[0])

    def dist_y(self, point1, point2):
        return abs(point2[1] - point1[1])
    
    def point_position(self, point, line_pt_1, line_pt_2):
        """
        Left or Right position of the point from a line
        Source: https://stackoverflow.com/a/62886424
        """
        value = (line_pt_2[0] - line_pt_1[0]) * (point[1] - line_pt_1[1]) - (line_pt_2[1] - line_pt_1[1]) * (point[0] - line_pt_1[0])
        if value >= 0:
            return "left"
        else:
            return "right"

