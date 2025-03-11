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

# Hough Transform
def hough_transform(image):
    height, width = image.shape
    max_dist = int(np.hypot(height, width))
    thetas = np.deg2rad(np.arange(0, 180))
    rhos = np.linspace(-max_dist, max_dist, max_dist * 2)

    accumulator = np.zeros((len(rhos), len(thetas)), dtype=np.int32)
    y_idxs, x_idxs = np.nonzero(image)
    for i in range(len(x_idxs)):
        x = x_idxs[i]
        y = y_idxs[i]
        for t_idx in range(len(thetas)):
            rho = int(x * np.cos(thetas[t_idx]) + y * np.sin(thetas[t_idx]) + max_dist)
            accumulator[rho, t_idx] += 1

    # Plot the accumulator array
    cv2.imshow('Accumulator', accumulator / np.max(accumulator))
    cv2.waitKey(0)

    # Non-maximum suppression
    threshold = 0.1 * np.max(accumulator)
    peaks = np.argwhere(accumulator > threshold)
    refined_peaks = np.zeros_like(accumulator)

    for peak in peaks:
        rho, theta = peak
        if accumulator[rho, theta] == np.max(accumulator[rho-1:rho+2, theta-1:theta+2]):
            refined_peaks[rho, theta] = accumulator[rho, theta]

    return refined_peaks

def draw(img, peaks):
    height, width = img.shape[:2]
    for peak in np.argwhere(peaks):
        rho, theta = peak
        a = np.cos(theta)
        b = np.sin(theta)
        x0 = a * rho
        y0 = b * rho
        x1 = int(x0 + 1000 * (-b))
        y1 = int(y0 + 1000 * (a))
        x2 = int(x0 - 1000 * (-b))
        y2 = int(y0 - 1000 * (a))
        cv2.line(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
    cv2.imshow('Lines', img)
    cv2.waitKey(0)
    
def main():
    test_img = cv2.imread('test.jpg')
    #if image too large, resize    
    blurried_img = median_blur(test_img)
    edge_img = canny_edge_detector(blurried_img)
    roi_img = get_ROI_edge(edge_img, test_img)  
    refined_peaks = hough_transform(roi_img)
    draw(test_img, refined_peaks)
    cv2.imshow('ROI', roi_img)
    cv2.imshow('Edge', edge_img)
    cv2.imshow('Blur', blurried_img)
    cv2.imshow('refined peaks', refined_peaks)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()