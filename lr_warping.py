import cv2 as cv
import numpy as np

def lr_warping(reference, lr_images, video_name):
    ref_keypoints, lr_keypoints_list, d_matches_list = feature_matching_operator(reference, lr_images)
    warped_images = []

    for i in range(len(lr_images)):
        warped_image = perform_warping(ref_keypoints, d_matches_list[i], lr_keypoints_list[i], lr_images[i])
        cv.imwrite("Temp/" + video_name + "_warped_" + str(i) + ".png", warped_image, [cv.IMWRITE_PNG_COMPRESSION, 0])

    # return warped_images


def perform_warping(ref_keypoints, good_match, candidate_keypoint, candidate_image):
    keypoints1 = np.asarray(ref_keypoints)
    keypoints2 = np.asarray(candidate_keypoint)

    point_list1 = []
    point_list2 = []

    match_array = np.asarray(good_match)

    for i in range(len(match_array)):
        point_list1.append(np.float32(keypoints1[match_array[i].queryIdx].pt))
        point_list2.append(np.float32(keypoints2[match_array[i].trainIdx].pt))
    
    mat_of_point1 = np.array(point_list1)
    mat_of_point2 = np.array(point_list2)
    
    homography = np.empty((1,1))
    if mat_of_point1.shape[0] > 0 and mat_of_point1.shape[1] > 0 and mat_of_point2.shape[0] > 0 and mat_of_point2.shape[1] > 0:
        homography, mask = cv.findHomography(mat_of_point2, mat_of_point1, cv.RANSAC)

    warped_image = cv.warpPerspective(candidate_image, homography, dsize=(candidate_image.shape[1], candidate_image.shape[0]) , flags=cv.INTER_LINEAR, borderMode= cv.BORDER_REPLICATE, borderValue = (0))

    return warped_image


def feature_matching_operator(reference, lr_images):
    ref_keypoints, ref_descriptor = detect_features_in_reference(reference)
    lr_keypoints_list = []
    d_matches_list = []


    for lr_image in lr_images:
        keypoints, matches = feature_matcher(ref_descriptor, lr_image)
        lr_keypoints_list.append(keypoints)
        d_matches_list.append(matches)
    
    return ref_keypoints, lr_keypoints_list, d_matches_list


def detect_features_in_reference(reference):
    orb = cv.ORB_create()
    keypoints = orb.detect(reference)
    keypoints, descriptor = orb.compute(reference, keypoints)


    return keypoints, descriptor

def feature_matcher(ref_descriptor, image_to_compare):
    orb = cv.ORB_create()
    keypoints = orb.detect(image_to_compare)
    keypoints, descriptor = orb.compute(image_to_compare, keypoints)
    matches = match_features_to_reference(ref_descriptor, descriptor)

    return keypoints, matches

def match_features_to_reference(ref_descriptor, descriptor):
    matcher = cv.BFMatcher(cv.NORM_HAMMING, crossCheck = True)
    initial_matches = matcher.match(ref_descriptor, descriptor)

    min_distance = 999.0
    good_matches_list = []

    for match in initial_matches:
        if match.distance < min_distance:
            good_matches_list.append(match)
    
    return good_matches_list