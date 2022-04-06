import cv2
import numpy as np
import os
import math
import logging
import time
from shapely.geometry import LineString, LinearRing
from skimage import img_as_bool, img_as_uint, morphology, color

input_dir = 'Input/'
img_number = 1
start_time = 0
filename = None
dst = None

#Produces outputs on every step.
debug_mode = False
#Applies color balance on the final output.
color_balance = True
#Adds an extra validation process when identifying the agar plate. This increases accuracy in some cases, but
validation_mode = False

def main():
    global img_number
    global masked_img
    global start_time
    global filename

    print("Initiating Image Processing")

    #Ref_img could be any underlighted agar plate.
    ref_img = cv2.imread("ref_img.png")
    initLog()
    file_array = sortInputFolder()
    for file_set in file_array:

        filename = file_set[0]
        start_time = time.perf_counter()

        if img_number >= 0 and cv2.imread(input_dir + filename) is not None:
            input_img = cv2.imread(input_dir + filename)
            margin_error = cv2.matchShapes(cv2.cvtColor(input_img, cv2.COLOR_BGR2GRAY), cv2.cvtColor(ref_img, cv2.COLOR_BGR2GRAY),cv2.CONTOURS_MATCH_I2, 0)

            #Compares the input to a reference to validate the input.
            if margin_error <= 0.2:
                if filename.endswith('.png'):
                    filename = filename[:-4]

                input_img = imageResize(input_img)  # Resize input to default 600x600px.
                input_img = imageBalance(input_img)  # Color, contrast, brightness balance.
                ellipse = identifyPlate(input_img.copy()) #Identify the agar plate
                if validation_mode:
                    ellipse = identifyPlateWithValidation(input_img.copy())
                masked_img = maskPlate(input_img.copy(), None, ellipse) #Mask agar plate using ellipse points.
                lines, center_point = findCompartmentEdges(masked_img.copy(), ellipse) #Find centerline of compartments dividers and centerpoint.
                key_points = identifyRotation(masked_img.copy(), ellipse, lines, center_point) #Segment red compartment, calculate intersection points and define rotation.
                imageRegistration(file_set, ellipse, key_points, dst)

                #####Logging#####
                end_time = time.perf_counter()
                print("Image set #" + str(img_number) + " " + filename + " processed successfully in: " + str(
                    round(end_time - start_time, 2)) + " seconds!")

                img_number += 1
                #logging.info(" ")
            else:
                print("Image set #" + str(img_number) + " " + filename  + " failed. See output log.")
                logging.warning("No agar plate found on image set: " + filename)
                logging.info(" ")
                img_number += 1
        else:
            img_number += 1
            logging.info("Image #" +  filename + " Failed!")
            logging.warning("Tried to read an unknown/wrong file: " + filename)
            logging.info(" ")

    os.startfile('output_log.log')


#Initiation
def initLog():
    if debug_mode:
        print("Debug mode activated:")
    if validation_mode:
        print("Validation mode activated:")
    if color_balance:
        print("Color balance activated:")
    if os.path.isfile('output_log.log'):
        os.remove('output_log.log')
    logging.basicConfig(filename='output_log.log', level=logging.DEBUG)
def sortInputFolder():
    decoded_list = []
    if os.listdir(input_dir) is None:
        logging.warning("Input folder: " + str(input_dir) + " not found.")
        return False

    for file in os.listdir(input_dir):

        filename = os.fsdecode(file)
        if len(filename.split('_')) == 5:
            decoded_list.append(filename)

    decoded_list.sort(key=lambda file: file.split('_')[4])

    ar = []
    file_array = []
    filename = decoded_list[0].split('_')[4]
    date = decoded_list[0].split('_')[0]
    for file in decoded_list:

        if file.split('_')[4] == filename and file.split('_')[0] == date:
            ar.append(file)
        else:
            file_array.append(ar)
            ar = []
            filename = file.split('_')[4]
            date = file.split('_')[0]
            ar.append(file)

    for i in file_array:
        i.sort(key=lambda file: file.split('_')[2], reverse=True)

    return file_array

#Image Balance
def imageBalance(img, balance_level = 1):
    img_bright = autoBrigthnessContrast(img.copy(), 0.2)
    img_color = simplestColorBalance(img_bright.copy(), balance_level)
    if debug_mode:
        cv2.imwrite("Debug/Correction/" + filename + "_color_balanced.png", img_color)
    return img_color
def autoBrigthnessContrast(img, clip_hist_percent):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Calculate grayscale histogram
    hist = cv2.calcHist([gray],[0],None,[256],[0,256])
    hist_size = len(hist)

    # Calculate cumulative distribution from the histogram
    accumulator = []
    accumulator.append(float(hist[0]))
    for index in range(1, hist_size):
        accumulator.append(accumulator[index -1] + float(hist[index]))

    # Locate points to clip
    maximum = accumulator[-1]
    clip_hist_percent *= (maximum/100.0)
    clip_hist_percent /= 2.0

    # Locate left cut
    minimum_gray = 0
    while accumulator[minimum_gray] < clip_hist_percent:
        minimum_gray += 1

    # Locate right cut
    maximum_gray = hist_size -1
    while accumulator[maximum_gray] >= (maximum - clip_hist_percent):
        maximum_gray -= 1

    # Calculate alpha and beta values
    alpha = 255 / (maximum_gray - minimum_gray)
    beta = -minimum_gray * alpha

    auto_result = cv2.convertScaleAbs(img, alpha=alpha, beta=0)

    if debug_mode:
        cv2.imwrite("Debug/Correction/" + filename + "_before.png", img)
        cv2.imwrite("Debug/Correction/" + filename + "_brigthness_balanced.png", auto_result)

    return auto_result
def simplestColorBalance(img, percent):
    assert img.shape[2] == 3
    assert percent > 0 and percent < 100

    half_percent = percent / 200.0

    channels = cv2.split(img)

    out_channels = []
    for channel in channels:
        assert len(channel.shape) == 2
        # find the low and high precentile values (based on the input percentile)
        height, width = channel.shape
        vec_size = width * height
        flat = channel.reshape(vec_size)

        assert len(flat.shape) == 1

        flat = np.sort(flat)

        n_cols = flat.shape[0]

        low_val  = flat[int(math.floor(n_cols * half_percent))]
        high_val = flat[int(math.ceil( n_cols * (1.0 - half_percent)))]

        # saturate below the low percentile and above the high percentile
        thresholded = apply_threshold(channel, low_val, high_val)
        # scale the channel
        normalized = cv2.normalize(thresholded, thresholded.copy(), 0, 255, cv2.NORM_MINMAX)
        out_channels.append(normalized)
        output = cv2.merge(out_channels)


    return cv2.merge(out_channels)

#Identifty Plate
def identifyPlate(img):
    lower_thresh = 10
    black = fillImage(height=600, width=600, color=0)
    imgray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_blur = cv2.GaussianBlur(imgray, (5, 5), 1)  # 5, 5
    img_canny_blur = autoCanny(img_blur, 1.5)

    edges = cv2.dilate(img_canny_blur, None, iterations=6)
    edges = cv2.erode(edges, None, iterations=6)
    _, thresh = cv2.threshold(edges, lower_thresh, 1500, type=cv2.THRESH_BINARY)

    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    sorted_list = sorted(contours, key=len, reverse=True)

    try:
        list_length = len(sorted_list[0])
    except IndexError:
        list_length = None

    while list_length is None or list_length < 650:
        if lower_thresh > 1500:
            break
        lower_thresh += 1

        _, thresh = cv2.threshold(imgray, lower_thresh, 1500, type=cv2.THRESH_BINARY)
        contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        sorted_list = sorted(contours, key=len, reverse=True)

    try:
        list_length = len(sorted_list[0])
    except IndexError:
        return None, None, 0

    cnt = sorted_list[0]

    ellipse = cv2.fitEllipse(cnt)  # ellipse = ((center),(width,height of bounding rect), angle)


    if debug_mode:
        match = cv2.drawContours(black.copy(), [cnt], 0, (0, 255, 0), 1)
        result = cv2.ellipse(img.copy(), ellipse, (0, 255, 0), 1)  # draw ellipse in red color
        cv2.imwrite('Debug/Ellipse/' + filename + '_edge.png', edges)
        cv2.imwrite('Debug/Ellipse/' + filename + '_contour.png', match)
        cv2.imwrite('Debug/Ellipse/' + filename + '_ellipse.png', result)
    return ellipse
def autoCanny(img, sigma):
    #img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    imgray = img
    #print("Auto canny")
    #blur = cv2.GaussianBlur(img, (5, 5), 0) # 5, 5
    blur = img
    v = np.median(blur)
    #sigma = 0.12

    lower = int(max(0, (1.0 - sigma) * v))
    upper = int(min(255, (1.0 + sigma) * v))
    edges = cv2.Canny(blur, lower, upper, apertureSize=3)
    #print("lower: " + str(lower))
    #print("upper: " + str(upper))
    return edges

#Identify Compartment Edges
def findCompartmentEdges(img, ellipse):
    processed = skeletonizeLines(img.copy(), ellipse)
    lines, center_point = houghLinesP(processed, img.copy())
    return lines, center_point
def skeletonizeLines(img, ellipse):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(img, 250, 255, type=cv2.THRESH_BINARY)
    thresh = cv2.dilate(thresh, None, iterations=3)
    thresh = cv2.erode(thresh, None, iterations=2)
    img_bool = img_as_bool(color.rgb2gray(thresh))
    out = morphology.skeletonize(img_bool)
    out = img_as_uint(out)

    #Mask the output
    mask = ((ellipse[0][0], ellipse[0][1]), (ellipse[1][0] / 2, ellipse[1][1] / 2), ellipse[2])
    out = cv2.cvtColor(out, cv2.COLOR_GRAY2BGR)
    out = maskPlate(out, None, mask)
    out = cv2.cvtColor(out, cv2.COLOR_BGR2GRAY)

    if debug_mode:
        cv2.imwrite('Debug/Lines/' + filename + '_edges.png', out)
    out = out.astype(np.uint8)

    return out
def houghLinesP(img, org):
    lines = cv2.HoughLinesP(img, 2, np.pi / 180, threshold=20, minLineLength=10, maxLineGap=250)
    img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    # Draw lines on the image
    line_coordinates = []
    idx = 0
    prev_theta = 45
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            theta = math.atan2(y2 - y1, x2 - x1)
            theta2 = math.atan2(y1 - y2, x1 - x2)
            a = np.cos(theta)
            b = np.sin(theta)
            if idx == 0 or round(abs(math.degrees(prev_theta) - (math.degrees(theta)))) >= 89:
                x1 = int(x1 + 1000 * a)
                y1 = int(y1 + 1000 * b)
                x2 = int(x2 - 1000 * a)
                y2 = int(y2 - 1000 * b)
                line_coordinates.append([(x1, y1), (x2, y2)])
                cv2.line(img, (x1, y1), (x2, y2), (255, 0, 0), 1)
                cv2.line(org, (x1, y1), (x2, y2), (255, 0, 0), 1)
                prev_theta = theta
                if len(line_coordinates) == 2:
                    break

            idx += 1

        if debug_mode:
            cv2.imwrite('Debug/Lines/' + filename + '_edges.png', img)
            cv2.imwrite('Debug/Lines/' + filename + '_houghlines.png', org)


        return line_coordinates, lineIntersection(line_coordinates[0], line_coordinates[1])

#Identify Rotation
def identifyRotation(img, ellipse, lines, center_point):
    global dst
    line1, line2 = sortLines(lines)
    ellipse_points, _ = ellipse_polyline(ellipse)
    intersection_pt = ellipseLineIntersection(LinearRing(ellipse_points), LineString(line1), LineString(line2))
    segmentation = findCompartment(img)
    landmark_pts = identifyOrientation(segmentation, intersection_pt)
    sorted_ellipse_pts = sortEllipsePoints(ellipse_points)
    sorted_ellipse_pts = sortArc(ellipse, sorted_ellipse_pts, landmark_pts)
    key_points = sortPoints(landmark_pts, sorted_ellipse_pts, center_point)

    if dst is None:
        dst = refPoints()

    return key_points
def sortLines(lines):

    min = lines[0][0][0] + lines[0][0][1]
    minline = None

    for line in lines:
        for x, y in line:
            if x + y < min:
                min = x + y
                minline = line

    if minline == lines[1]:
        lines[1] = lines[0]
        lines[0] = minline

    return lines[0], lines[1]
def ellipse_polyline(ellipses, n=5000):
    t = np.linspace(0, 2 * np.pi, n, endpoint=False)
    st = np.sin(t)
    ct = np.cos(t)
    result = []
    ellipses = [ellipses]
    x0, y0 = ellipses[0][0]
    a, b = ellipses[0][1]
    angle = ellipses[0][2]
    ellipses = [[x0, y0, a, b, angle], [x0, y0, a, b, angle]]
    for x, y, a1, b2, ang in ellipses:
        ang = np.deg2rad(ang)
        sa = np.sin(ang)
        ca = np.cos(ang)
        p = np.empty((n, 2))
        a1 = a1 / 2
        b2 = b2 / 2
        p[:, 0] = x + a1 * ca * ct - b2 * sa * st
        p[:, 1] = y + a1 * sa * ct + b2 * ca * st
        result.append(p)
    return result
def ellipseLineIntersection(ellipse_points, line1, line2):
    l1 = ellipse_points.intersection(line1)
    l2 = ellipse_points.intersection(line2)

    x = [round(p.x) for p in l1]
    y = [round(p.y) for p in l1]
    x2 = [round(p.x) for p in l2]
    y2 = [round(p.y) for p in l2]

    points = [(x[0], y[0]), (x2[1], y2[1]), (x[1], y[1]), (x2[0], y2[0])] #Right order from (0, 0) -> (1, 0) -> (1, 1) -> (0, 1)
    return points
def findCompartment(img):
    if img is not None:
        output = img
        # Set minimum and max HSV values to display
        lower = np.array([0, 130, 0]) #0, 50, 0 # VALUES TO FIND RED
        upper = np.array([179, 255, 255]) #179, 255, 255 # VALUES TO FIND RED

        # Create HSV Image and threshold into a range.
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, lower, upper)
        output = cv2.bitwise_and(img, img, mask=mask)

        # Display output image
        if debug_mode:
            cv2.imwrite('Debug/Find Red/' + filename + '_segmentation.png', output)
        return output
    else:
        logging.warning("Red segmentation failed: Image appears to be empty")
def identifyOrientation(img, intersection_pt):
    intersection_pt.append(intersection_pt[0])
    sorted_points = []
    idx = 0;
    highest_mean_value = 0
    for i in range(0, len(intersection_pt)-1):
        if i != len(intersection_pt):
            value = 0
            line = sortLinePoints([intersection_pt[i], intersection_pt[i+1]])
            for x,y in line:
                bgr_pixel_value = img[y, x]
                value += bgr_pixel_value[2]

            mean_value = value/len(line)
            if mean_value > highest_mean_value:
                highest_mean_value = mean_value
                idx = i

    sorted_points.append(intersection_pt[idx])
    sorted_points.append(intersection_pt[idx+1])

    for i in range(0, len(intersection_pt) - 1):
        if i != len(intersection_pt):
            line = [intersection_pt[i], intersection_pt[i+1]]
            if sorted_points[0] not in line and sorted_points[1] not in line:
                sorted_points.append(intersection_pt[i])
                sorted_points.append(intersection_pt[i+1])
                break
    return sorted_points
def sortEllipsePoints(ellipse_points):
    ls = LinearRing(ellipse_points)
    steps = ls.length / len(ellipse_points)
    xy = []

    for f in np.arange(0, ls.length, steps):
        p = ls.interpolate(f).coords[0]
        if p not in xy:
            xy.append(p)

    ar = np.array(xy, 'f')
    return ar
def sortArc(ellipse, ellipse_points, landmark_pts):
    center_x, center_y = ellipse[0]
    landmark_angle = math.degrees(math.atan2(center_y - landmark_pts[0][1] , center_x - landmark_pts[0][0])) % 360
    count = 0
    black = fillImage(600, 600, 0)
    step = 0
    ellipse_degree = {}
    for point in ellipse_points:
        x1 = point[0]
        y1 = point[1]
        degree = math.degrees(math.atan2(center_y - y1 , center_x - x1)) % 360

        if degree - landmark_angle < 0:
            degree = 360 - abs(degree - landmark_angle)

        else:
            degree -= landmark_angle
        step +=1
        ellipse_degree[(x1, y1)] = degree
        if step % 10 == 0:
            count += 1
            cv2.circle(black, (int(x1), int(y1)), 2, (0, 0, 255), 3)
            cv2.putText(black, 'a:{}'.format(round(degree)), (int(x1), int(y1)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    sorted_points = sorted(ellipse_degree.items(), key=lambda kv: kv[1])

    if debug_mode:
        cv2.imwrite('Debug/Intersection/' + filename + '_sorted.png', black)

    return sorted_points
def sortPoints(landmark_pts, sorted_ellipse_pts, center_point):

    key_points = []
    line1_points = sortLinePoints([landmark_pts[0], landmark_pts[2]])
    line2_points = sortLinePoints([landmark_pts[1], landmark_pts[3]])

    for i in landmark_pts:
        key_points.append(i)

    key_points.append(center_point)

    for i in line1_points:
        key_points.append(i)

    for i in line2_points:
        key_points.append(i)

    for i in sorted_ellipse_pts:
        key_points.append(i[0])

    if debug_mode:
        black = fillImage(600, 600, 0)
        count = 0
        for i in key_points:
            count += 1
            if count <= 5:
                cv2.circle(black, (int(i[0]), int(i[1])), 2, (0, 0, 255), 3)
                cv2.putText(black, '{}'.format(' ' + str(count)), (int(i[0]) + 2, int(i[1])), cv2.FONT_HERSHEY_SIMPLEX, 0.3,
                            (0, 0, 255),
                            3)
            if count > 5:
                cv2.circle(black, (int(i[0]), int(i[1])), 2, (0, 255, 0), 3)
                cv2.putText(black, '{}'.format(' ' + str(count)), (int(i[0])+2, int(i[1])), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 255, 0), 3)
        cv2.imwrite('Debug/Intersection/' + filename + '_intersection.png', black)

    key_points = np.float32(key_points)
    return key_points
def sortLinePoints(line):
    no_of_points = 2500
    ls = LineString(line)
    parts = split_equal(ls.length, no_of_points)
    steps = getMean(parts)

    xy = []

    for f in np.arange(steps, ls.length-steps, steps):
        p = ls.interpolate(f).coords[0]

        if p not in xy:
            xy.append(p)

    ar = np.array(xy, 'i')

    return ar
def split_equal(value, parts):
    value = float(value)
    return [i*value/parts for i in range(1,parts+1)]
def refPoints():

    # Destination points fixed to center 600x600 image.
    points = []
    dst_ellipse = ((300, 300), (500, 500), 90)
    dst_line1 = [(600, 0), (0, 600)]
    dst_line2 = [(600, 600), (0, 0)]


    dst_ellipse, _ = ellipse_polyline(dst_ellipse)
    dst_line1, dst_line2 = sortLines([dst_line1, dst_line2])
    dst = ellipseLineIntersection(LinearRing(dst_ellipse), LineString(dst_line1), LineString(dst_line2))

    line1_points = sortLinePoints([dst[1], dst[3]])
    line2_points = sortLinePoints([dst[2], dst[0]])
    dst = [dst[1], dst[2], dst[3], dst[0]]
    ellipse_points = sortEllipsePoints(dst_ellipse)
    ellipse_points = sortArc(dst_ellipse, ellipse_points, dst)

    count = 0
    black = fillImage(600, 600, 0)
    for i in dst:
        count += 1
        points.append(i)
        cv2.circle(black, (round(i[0]), round(i[1])), 2, (0, 0, 255), 3)
        cv2.putText(black, '{}'.format(' ' + str(count)), (int(i[0]) + 2, int(i[1])), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255),
                   2)

    points.append((300, 300))
    for i in line1_points:
        points.append(i)

    for i in line2_points:
        points.append(i)

    for i in ellipse_points:
       points.append(i[0])


    count = 0
    black = fillImage(600, 600, 0)
    for i in points:
        count += 1
        if count <= 5:
            cv2.circle(black, (round(i[0]), round(i[1])), 2, (0, 0, 255), 3)
            cv2.putText(black, '{}'.format(' ' + str(count)), (int(i[0]) + 2, int(i[1])), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        (0, 0, 255),
                        2)
        if count > 5:
            cv2.circle(black, (int(i[0]), int(i[1])), 2, (0, 255, 0), 3)
            cv2.putText(black, '{}'.format(' ' + str(count)), (int(i[0]) + 2, int(i[1])), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        (0, 255, 0), 2)


    points = np.float32(points)

    return points

#Image Registration
def imageRegistration(file_set, ellipse, key_points, dst):

    for file in file_set:
        img = cv2.imread(input_dir + file)

        if img is not None:
            img = imageResize(img, 600, 600)
            img = maskPlate(img.copy(), None, ellipse)
            img, _ = unwarp(img, key_points, dst)

            if color_balance:
                img = imageBalance(img, 0.5)
            cv2.imwrite('Output/' + file, img)
        else:
            logging.warning("Unsupported format or broken file: " + str(file))
def unwarp(img, src, dst):

    h, w = img.shape[:2]

    if(len(src) != len(dst)):
        logging.warning("Error! SRC != DST" + " Source: " + str(len(src) + " | Dest: " + str(len(dst))))
        return None, None
    else:
        H, status = cv2.findHomography(src, dst)
        warped = cv2.warpPerspective(img.copy(), H, (w, h), flags=cv2.WARP_FILL_OUTLIERS + cv2.INTER_LANCZOS4)


        if debug_mode:
            cv2.imwrite('Debug/Warped/' + filename + '_warped.png', warped)
            cv2.imwrite('Debug/Warped/' + filename + '_unwarped.png', img)
        return warped, H

#Other
def maskPlate(img, circles = None, ellipse = None):
    mask = np.zeros((img.shape[1], img.shape[0], 3), dtype=np.uint8)
    out = img

    if circles is not None:
        circles = np.round(circles[0, :]).astype("int")
        if len(circles) == 1:
            x, y, r = circles[0]
            cv2.circle(mask, (x, y), r, (255, 255, 255), -1, 8, 0)
            mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
            out = cv2.bitwise_and(img, img, mask=mask)

    elif ellipse is not None:
        ellipse_shape = cv2.ellipse(fillImage(600, 600, 0), ellipse, (255, 255, 255), -1)  # draw ellipse in red color
        ellipse_shape = cv2.cvtColor(ellipse_shape, cv2.COLOR_BGR2GRAY)
        out = cv2.bitwise_and(img, img, mask=ellipse_shape)

    else:
        logging.warning('No circles or ellipse found to proceed masking')

    if debug_mode:
        cv2.imwrite('Debug/Mask/' + filename + '_crop_black.png', out)

    #cv2.imshow()
    return out
def lineIntersection(p1, p2):
    line1 = LineString(p1)
    line2 = LineString(p2)
    int_pt = line1.intersection(line2)
    point_of_intersection = int_pt.x, int_pt.y

    return point_of_intersection
def imageResize(img, height=600, width=600):
    if img is not None:
        dim = (height, width)
        # resize image
        img = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)
    else:
        logging.warning("Image resize failed: Image seems to be empty")
    return img
def fillImage(height, width, color):
    black = np.zeros([height, width, 3], dtype=np.uint8)
    black.fill(color)
    return black
def map(x, in_min, in_max, out_min, out_max):
    return int((x - in_min) * (out_max - out_min) / (in_max - in_min) + out_min)
def getMean(parts):
    mean = 0
    for i in range(0, len(parts)-1):
        value = parts[i+1] - parts[i]
        mean += value
    return mean/len(parts)

#Only used in validation_mode. An additional validation is used when finding the contour. If a contour is too far away from it circular representation (HoughCircle). Brigthness and contrast are automatically adjusted, and another iteration starts.
def apply_mask(matrix, mask, fill_value):
    masked = np.ma.array(matrix, mask=mask, fill_value=fill_value)
    return masked.filled()
def apply_threshold(matrix, low_value, high_value):
    low_mask = matrix < low_value
    matrix = apply_mask(matrix, low_mask, low_value)

    high_mask = matrix > high_value
    matrix = apply_mask(matrix, high_mask, high_value)

    return matrix
def apply_brightness_contrast(input_img, brightness=255, contrast=127):
    brightness = map(brightness, 0, 510, -255, 255)
    contrast = map(contrast, 0, 254, -127, 127)

    if brightness != 0:
        if brightness > 0:
            shadow = brightness
            highlight = 255
        else:
            shadow = 0
            highlight = 255 + brightness
        alpha_b = (highlight - shadow) / 255
        gamma_b = shadow

        buf = cv2.addWeighted(input_img, alpha_b, input_img, 0, gamma_b)
    else:
        buf = input_img.copy()

    if contrast != 0:
        if contrast == 131:
            contrast -= 1
        f = float(131 * (contrast + 127)) / (127 * (131 - contrast))
        alpha_c = f
        gamma_c = 127 * (1 - f)

        buf = cv2.addWeighted(buf, alpha_c, buf, 0, gamma_c)

    return buf
def identifyPlateWithValidation(img):
    brightness = 250
    contrast = 132
    lower_thresh = 10
    M = 0
    while M < 0.80:
        black = fillImage(height=600, width=600, color=0)
        imgray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        circle, ref = houghCircle(img.copy())
        img_blur = cv2.GaussianBlur(imgray, (5, 5), 1) # 5, 5
        img_canny_blur = autoCanny(img_blur, 1.5)

        edges = cv2.dilate(img_canny_blur, None, iterations=6)
        edges = cv2.erode(edges, None, iterations=6)
        _, thresh = cv2.threshold(edges, lower_thresh, 1500, type=cv2.THRESH_BINARY)

        contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

        sorted_list = sorted(contours, key=len, reverse=True)
        try:
            list_length = len(sorted_list[0])
        except IndexError:
            list_length = None

        while list_length is None or list_length < 650:
            if lower_thresh > 1500:
                break
            lower_thresh += 1

            _, thresh = cv2.threshold(imgray, lower_thresh, 1500, type=cv2.THRESH_BINARY)
            contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            sorted_list = sorted(contours, key=len, reverse=True)

        try:
            list_length = len(sorted_list[0])
        except IndexError:
            return None, None, 0

        cnt = sorted_list[0]

        match = cv2.drawContours(black.copy(), [cnt], 0, (0, 255, 0), 1)  # draw contours in green color

        # Check matching
        match = cv2.cvtColor(match, cv2.COLOR_BGR2GRAY)
        ref = cv2.cvtColor(ref, cv2.COLOR_BGR2GRAY)
        M = cv2.matchShapes(match, ref, cv2.CONTOURS_MATCH_I2, 0)


        if M < 0.80:
            if debug_mode:
                cv2.imwrite('Debug/Failed/' + filename + '_badcontour.png', match)
                cv2.imwrite('Debug/Ellipse/' + filename + '_contour.png', match)
            contrast += 1
            brightness -= 1
            img = apply_brightness_contrast(img.copy(), brightness, contrast)


        else:
            ellipse = cv2.fitEllipse(cnt)  # ellipse = ((center),(width,height of bounding rect), angle)
            # Draw contour, poly, and ellipse
            result = cv2.ellipse(img.copy(), ellipse, (0, 255, 0), 1)  # draw ellipse in red color
            if debug_mode:
                cv2.imwrite('Debug/Ellipse/' + filename + '_edge.png', edges)
                cv2.imwrite('Debug/Ellipse/' + filename + '_contour.png', match)
                cv2.imwrite('Debug/Ellipse/' + filename + '_ellipse.png', result)
            return ellipse
def houghCircle(input_img):
    gray_img = cv2.cvtColor(input_img, cv2.COLOR_BGR2GRAY)
    gray_blurred = cv2.blur(gray_img, (5, 5))

    # Black used for matchShape
    black = fillImage(height=600, width=600, color=0)
    img = cv2.medianBlur(gray_blurred, 5)
    edges = cv2.dilate(img, None, iterations=1)
    circles = cv2.HoughCircles(edges, cv2.HOUGH_GRADIENT, 1, 1500, param1=3, param2=1, minRadius=245, maxRadius=270)

    # Changes the center point of the circle, to the houghline intersection
    '''if center_point is not None:
        x, y = center_point
        circles[0][0][0] = x
        circles[0][0][1] = y'''

    if circles is not None:
        circles = np.uint16(np.around(circles))

        for i in circles[0, :]:
            # draw the outer circle
            cv2.circle(input_img, (i[0], i[1]), i[2], (0, 255, 0), 1)
            cv2.circle(black, (i[0], i[1]), i[2], (0, 255, 0), 5)
            # draw the center of the circle
            center = cv2.circle(input_img, (i[0], i[1]), 2, (0, 255, 0), 3)
    else:
        logging.warning("No circle found on: " + filename)

    if debug_mode:
        output = 'Debug/Hough/' + filename + '_houghCircle.png'
        cv2.imwrite(output, input_img)

    center_point = [circles[0][0][0], circles[0][0][1]]

    return circles, black #


if __name__ == "__main__":
    main()
