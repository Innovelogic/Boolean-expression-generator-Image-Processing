import math
from lines_recognition import LinesRecognition as lrec
import cv2


class LineReductions:

    @staticmethod
    def get_lines(lines_in):
        if cv2.__version__ < '3.0':
            return lines_in[0]
        return [l[0] for l in lines_in]

    @staticmethod
    def merge_lines_pipeline_2(lines):
        super_lines_final = []
        super_lines = []
        min_distance_to_merge = 30
        min_angle_to_merge = 30

        for line in lines:
            create_new_group = True
            group_updated = False

            for group in super_lines:
                for line2 in group:
                    if LineReductions.get_distance(line2, line) < min_distance_to_merge:
                        # check the angle between lines
                        orientation_i = math.atan2((line[0][1] - line[1][1]), (line[0][0] - line[1][0]))
                        orientation_j = math.atan2((line2[0][1] - line2[1][1]), (line2[0][0] - line2[1][0]))

                        if int(abs(
                                abs(math.degrees(orientation_i)) - abs(math.degrees(orientation_j)))) < min_angle_to_merge:
                            # print("angles", orientation_i, orientation_j)
                            # print(int(abs(orientation_i - orientation_j)))
                            group.append(line)

                            create_new_group = False
                            group_updated = True
                            break

                if group_updated:
                    break

            if (create_new_group):
                new_group = []
                new_group.append(line)

                for idx, line2 in enumerate(lines):
                    # check the distance between lines
                    if LineReductions.get_distance(line2, line) < min_distance_to_merge:
                        # check the angle between lines
                        orientation_i = math.atan2((line[0][1] - line[1][1]), (line[0][0] - line[1][0]))
                        orientation_j = math.atan2((line2[0][1] - line2[1][1]), (line2[0][0] - line2[1][0]))

                        if int(abs(
                                abs(math.degrees(orientation_i)) - abs(math.degrees(orientation_j)))) < min_angle_to_merge:
                            # print("angles", orientation_i, orientation_j)
                            # print(int(abs(orientation_i - orientation_j)))

                            new_group.append(line2)

                            # remove line from lines list
                            # lines[idx] = False
                # append new group
                super_lines.append(new_group)

        for group in super_lines:
            super_lines_final.append(LineReductions.merge_lines_segments1(group))

        return super_lines_final

    @staticmethod
    def merge_lines_segments1(lines, use_log=False):
        if (len(lines) == 1):
            return lines[0]

        line_i = lines[0]

        # orientation
        orientation_i = math.atan2((line_i[0][1] - line_i[1][1]), (line_i[0][0] - line_i[1][0]))

        points = []
        for line in lines:
            points.append(line[0])
            points.append(line[1])

        if (abs(math.degrees(orientation_i)) > 45) and abs(math.degrees(orientation_i)) < (90 + 45):

            # sort by y
            points = sorted(points, key=lambda point: point[1])

            if use_log:
                print("use y")
        else:

            # sort by x
            points = sorted(points, key=lambda point: point[0])

            if use_log:
                print("use x")

        return [points[0], points[len(points) - 1]]

    @staticmethod
    def lines_close(line1, line2):
        dist1 = math.hypot(line1[0][0] - line2[0][0], line1[0][0] - line2[0][1])
        dist2 = math.hypot(line1[0][2] - line2[0][0], line1[0][3] - line2[0][1])
        dist3 = math.hypot(line1[0][0] - line2[0][2], line1[0][0] - line2[0][3])
        dist4 = math.hypot(line1[0][2] - line2[0][2], line1[0][3] - line2[0][3])
        if (min(dist1, dist2, dist3, dist4) < 100):
            return True
        else:
            return False

    @staticmethod
    def lineMagnitude(x1, y1, x2, y2):
        lineMagnitude = math.sqrt(math.pow((x2 - x1), 2) + math.pow((y2 - y1), 2))
        return lineMagnitude



    @staticmethod
    def DistancePointLine(px, py, x1, y1, x2, y2):

        LineMag = LineReductions.lineMagnitude(x1, y1, x2, y2)

        if LineMag < 0.00000001:
            DistancePointLine = 9999
            return DistancePointLine

        u1 = (((px - x1) * (x2 - x1)) + ((py - y1) * (y2 - y1)))
        u = u1 / (LineMag * LineMag)

        if (u < 0.00001) or (u > 1):
            # // closest point does not fall within the line segment, take the shorter distance
            # // to an endpoint
            ix = LineReductions.lineMagnitude(px, py, x1, y1)
            iy = LineReductions.lineMagnitude(px, py, x2, y2)
            if ix > iy:
                DistancePointLine = iy
            else:
                DistancePointLine = ix
        else:
            # Intersecting point is on the line, use the formula
            ix = x1 + u * (x2 - x1)
            iy = y1 + u * (y2 - y1)
            DistancePointLine = LineReductions.lineMagnitude(px, py, ix, iy)

        return DistancePointLine

    @staticmethod
    def get_distance(line1, line2):
        dist1 = LineReductions.DistancePointLine(line1[0][0], line1[0][1],
                                  line2[0][0], line2[0][1], line2[1][0], line2[1][1])
        dist2 = LineReductions.DistancePointLine(line1[1][0], line1[1][1],
                                  line2[0][0], line2[0][1], line2[1][0], line2[1][1])
        dist3 = LineReductions.DistancePointLine(line2[0][0], line2[0][1],
                                  line1[0][0], line1[0][1], line1[1][0], line1[1][1])
        dist4 = LineReductions.DistancePointLine(line2[1][0], line2[1][1],
                                  line1[0][0], line1[0][1], line1[1][0], line1[1][1])

        return min(dist1, dist2, dist3, dist4)


    @staticmethod
    def merge_lines_logic():
        merged_lines_all = []
        lines = lrec.grab_lines()

        _lines = []
        for _line in LineReductions.get_lines(lines):
            _lines.append([(_line[0], _line[1]), (_line[2], _line[3])])

        # sort
        _lines_x = []
        _lines_y = []
        for line_i in _lines:
            orientation_i = math.atan2((line_i[0][1] - line_i[1][1]), (line_i[0][0] - line_i[1][0]))
            if (abs(math.degrees(orientation_i)) > 45) and abs(math.degrees(orientation_i)) < (90 + 45):
                _lines_y.append(line_i)
            else:
                _lines_x.append(line_i)

        _lines_x = sorted(_lines_x, key=lambda _line: _line[0][0])
        _lines_y = sorted(_lines_y, key=lambda _line: _line[0][1])

        merged_lines_x = LineReductions.merge_lines_pipeline_2(_lines_x)
        merged_lines_y = LineReductions.merge_lines_pipeline_2(_lines_y)

        merged_lines_all.extend(merged_lines_x)
        merged_lines_all.extend(merged_lines_y)


        return merged_lines_all
