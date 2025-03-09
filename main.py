import cv2 as cv
import numpy as np

def median_blur(img):
    return cv.medianBlur(img, 5)

def canny_edge_detector(img):
    return cv.Canny(img, 100, 200)

def get_ROI_edge(edge_img, img):
    height, width = edge_img.shape[:2]

    # Define points for the quadrilateral ROI
    roi_points = np.array([
        [0, height],  # Bottom-left corner
        [width, height],  # Bottom-right corner
        [width * 2 // 3, height * 2 // 5],  # Right middle of upper third
        [width // 3, height * 2 // 5]  # Left middle of upper third
    ], dtype=np.int32)
    
    # Create a mask with the same dimensions as the image
    mask = np.zeros_like(edge_img)
    
    # Fill the ROI in the mask with white color
    cv.fillPoly(mask, [roi_points], 255)
    
    # Apply the ROI mask to the edge image
    masked = cv.bitwise_and(edge_img, mask)
    
    # Draw the points of interest on the original image
    img_with_points = cv.cvtColor(edge_img, cv.COLOR_GRAY2BGR)  # Convert to BGR for drawing
    for point in roi_points:
        cv.circle(img_with_points, tuple(point), 5, (0, 255, 0), -1)
    
    cv.imshow('Points', img_with_points)
    cv.imshow('Masked', masked)
    
    return masked

def hough_transform(image):
    # accumulation into ro, theta space

    # refining coordinates
    pass
    
def main():
    test_img = cv.imread('3.jpg')
    blurried_img = median_blur(test_img)
    edge_img = canny_edge_detector(blurried_img)
    roi_img = get_ROI_edge(edge_img, test_img)
    cv.imshow('ROI', roi_img)
    cv.imshow('Edge', edge_img)
    cv.imshow('Blur', blurried_img)
    cv.waitKey(0)
    cv.destroyAllWindows()

if __name__ == '__main__':
    main()