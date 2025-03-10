import cv2
import numpy as np

def median_blur(img):
    return cv2.medianBlur(img, 7)

def canny_edge_detector(img):
    return cv2.Canny(img, 100, 200)


def get_ROI_edge(edge_img, img):
    height, width = edge_img.shape[:2]

    # Define points for the quadrilateral ROI
    roi_points = np.array([
        [width, height],  # Bottom-right corner
        [0, height],  # Bottom-left corner
        [0, height - 160],  # Bottom-left corner
        [width * 1 // 4, height * 4 // 9],  # Right middle of upper third
        [width * 2 // 4, height * 4 // 9]  # Left middle of upper third
    ], dtype=np.int32)
    
    # Create a mask with the same dimensions as the image
    mask = np.zeros_like(edge_img)
    
    # Fill the ROI in the mask with white color
    cv2.fillPoly(mask, [roi_points], 255)
    
    # Apply the ROI mask to the edge image
    masked = cv2.bitwise_and(edge_img, mask)
    masked_img = cv2.bitwise_and(img, img, mask=mask)
    
    # Draw the points of interest on the original image
    img_with_points = cv2.cvtColor(edge_img, cv2.COLOR_GRAY2BGR)  # Convert to BGR for drawing
    for point in roi_points:
        cv2.circle(img_with_points, tuple(point), 5, (0, 255, 0), -1)
    
    cv2.imshow('Points', img_with_points)
    cv2.imshow('Masked', masked)
    cv2.imshow('Masked Image', masked_img)
    
    return masked

def hough_transform(image):
    # accumulation into ro, theta space

    # refining coordinates
    pass
    
def main():
    test_img = cv2.imread('test.jpg')
    #if image too large, resize    
    blurried_img = median_blur(test_img)
    edge_img = canny_edge_detector(blurried_img)
    roi_img = get_ROI_edge(edge_img, test_img)
    cv2.imshow('ROI', roi_img)
    cv2.imshow('Edge', edge_img)
    cv2.imshow('Blur', blurried_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()