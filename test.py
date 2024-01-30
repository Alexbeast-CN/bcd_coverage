import cv2
import numpy as np
import matplotlib.pyplot as plt

def find_interesting_contour(gray_map, current_pos):
    contours, hierarchy = cv2.findContours(gray_map, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    minContour = None
    allChildContours = []
    minArea = None
    
    index = 0
    for i, contour in enumerate(contours):
        if cv2.pointPolygonTest(contour, current_pos, False) >= 0:
            area = cv2.contourArea(contour)
            if minContour is None or area < minArea:
                minArea = area
                minContour = contour
                index = i
                
    def find_all_child_contours(contours, hierarchy, parentIdx, childContours):
        if parentIdx < 0 or parentIdx >= len(hierarchy[0]):
            return
        childIdx = hierarchy[0][parentIdx][2]
        while childIdx != -1:
            childContours.append(contours[childIdx])
            find_all_child_contours(contours, hierarchy, childIdx, childContours)
            childIdx = hierarchy[0][childIdx][0]
    
    find_all_child_contours(contours, hierarchy, index, allChildContours)
    selected_contours = [minContour] + allChildContours
    def merge_close_points(contour, distance_threshold):
        """
        Merge points in a contour that are closer than the specified distance threshold.
        """
        if len(contour) < 2:
            return contour

        merged_contour = []
        skip_next = False
        for i in range(len(contour)):
            if skip_next:
                skip_next = False
                continue

            if i == len(contour) - 1:
                merged_contour.append(contour[i])
            else:
                current_point = contour[i][0]
                next_point = contour[i+1][0]
                distance = np.linalg.norm(current_point - next_point)

                if distance < distance_threshold:
                    merged_point = ((current_point + next_point) / 2).astype(np.int32)
                    merged_contour.append([merged_point])
                    skip_next = True
                else:
                    merged_contour.append(contour[i])

        return np.array(merged_contour)
    
    # Merge close points in contours
    distance_threshold = 3
    merged_contours = [merge_close_points(cnt, distance_threshold) for cnt in selected_contours]
    
    # simplify the contours
    simplified_contours = []
    for cnt in merged_contours:
        epsilon = 0.02 * cv2.arcLength(cnt, True)  # Increased epsilon for more simplification
        simplified_contour = cv2.approxPolyDP(cnt, epsilon, True)
        simplified_contours.append(simplified_contour)
        
    return simplified_contours

# Load the image
file_path = 'grid_map2.pgm'
image = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)

# convert the image to binary
ret, binary = cv2.threshold(image, 200, 255, cv2.THRESH_BINARY)

# errode the map 
erode_width = 5
erode_height = 5
element = cv2.getStructuringElement(cv2.MORPH_RECT, (erode_width, erode_height))
inflated_map = cv2.erode(binary, element)

# Process the image and find contours
contours = find_interesting_contour(inflated_map, (100, 100))

for cnt in contours:
    print("Number of points in contour: ", len(cnt))

# Visualize the results
contour_board = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
plt.figure(figsize=(10, 5))
cv2.drawContours(contour_board, contours, -1, (0, 255, 0), 1)
plt.imshow(contour_board, cmap='gray')
plt.title("Original Contours")
plt.show()

