import cv2
import numpy as np
from typing import Union
from image_proccesing import ImageProcessing
from scipy.spatial import distance
import mahotas as mt
from skimage import feature
from scipy.stats import mode


class FeatureExtraction(ImageProcessing):
    def __init__(self, image_path: str, image_mask_path: str):
        super().__init__(image_mask_path)
        self.image = cv2.imread(image_path)

    def get_circle_mask(self, radius: float, coef: float = 1.0) -> np.ndarray:
        mask = np.zeros(self.mask_shape)
        cv2.circle(mask, self.centroid, int(coef * radius), 125, -1)
        return mask

    def get_ellipse_mask(self, center_x: int, center_y: int, s_half_ax: float, b_half_ax: float, angle: int, coef: float = 1.0) -> np.ndarray:
        mask = np.zeros(self.mask_shape)
        cv2.ellipse(mask, (center_x, center_y), (int(coef * b_half_ax), int(coef * s_half_ax)), angle, 0, 360, 125, -1)
        return mask

    def get_hull_mask(self, hull: np.ndarray, coef: float = 1.0) -> np.ndarray:
        mask = np.zeros(self.mask_shape)
        if coef != 1.0:
            hull = self.centroid + coef * (hull - self.centroid)
            hull = hull.astype(int)
        cv2.fillPoly(mask, [hull], 125)
        return mask

    def get_area_feature(self, channel: Union[int, str], threshold: int, feature_mask: np.ndarray) -> float:
        cleaned_image_mask = cv2.fillPoly(np.zeros(self.mask_shape), self.cleaned_contours, 125)
        result_mask = np.bitwise_and(cleaned_image_mask == 125, feature_mask == 125).astype('uint8') * 255
        result_image = cv2.bitwise_and(self.image, self.image, mask=result_mask)
        mask_area = np.sum(result_mask > 0)
        if type(channel) is int:
            image_area = np.sum(cv2.threshold(result_image[:, :, channel], threshold, 255, cv2.THRESH_BINARY)[1] > 0)
        else:
            image_area = np.sum(cv2.threshold(result_image, threshold, 255, cv2.THRESH_BINARY)[1] > 0)
        if mask_area == 0.0:
            # print('mask_area equal 0.0')
            return 0.0
        return np.round(image_area / mask_area, 6)

    def get_exact_area_feature(self, channel: Union[int, str], threshold: int, mask_type: str, coef: float = 1.0):
        if mask_type == 'ellipse':
            feature_mask = self.get_ellipse_mask(*self.get_ellipse_params(), coef=coef)
        elif mask_type == 'circle':
            feature_mask = self.get_circle_mask(self.get_circle_radius(), coef=coef)
        elif mask_type == 'hull':
            feature_mask = self.get_hull_mask(self.convex_hull, coef=coef)
        else:
            return 'wrong mask type'
        return self.get_area_feature(channel, threshold, feature_mask)

    @staticmethod
    def statistics_for_contour_features(values):
        return [np.min(values), np.round(np.mean(values), 6), np.max(values)]

    def get_contour_features(self):
        mask = np.zeros(self.mask_shape)
        mask = cv2.fillPoly(mask, [self.convex_hull], 255)
        mask_area = np.sum(mask == 255)
        bitwise_area_to_convex_hull = np.sum((np.bitwise_and(self.image_mask == 255, mask == 255))) / mask_area

        total_area_to_convex_hull = np.sum(self.image_mask == 255) / mask_area

        contour_area_to_convex_hull = []
        img_area_to_convex_hull = []
        img_area_to_approx_hull = []

        contour_roughness_1 = []
        contour_roughness_6 = []

        min_rad = []
        mean_rad = []

        radial_deviation = []

        convexity_changes = 0

        for cnt in self.cleaned_contours:
            mask = np.zeros(self.mask_shape)
            img_area = np.sum(np.bitwise_and(self.image_mask == 255, cv2.fillPoly(mask, [cnt], 255) == 255))
            cnt_area = cv2.contourArea(cnt)
            convex_hull_contour_area = cv2.contourArea(cv2.convexHull(cnt))

            contour_area_to_convex_hull.append(cnt_area / convex_hull_contour_area)
            img_area_to_convex_hull.append(img_area / convex_hull_contour_area)

            epsilon = 0.005 * cv2.arcLength(cnt, True)
            approx_hull = cv2.approxPolyDP(cnt, epsilon, True)

            prev_convexity = None
            for i in range(len(approx_hull)):
                p1 = approx_hull[i][0]
                p2 = approx_hull[(i + 1) % len(approx_hull)][0]
                p3 = approx_hull[(i + 2) % len(approx_hull)][0]
                cross_product = np.cross(p1 - p2, p3 - p2)

                # Определение выпуклости или вогнутости
                current_convexity = cross_product > 0

                if prev_convexity is not None and current_convexity != prev_convexity:
                    convexity_changes += 1

                prev_convexity = current_convexity

            img_area_to_approx_hull.append(
                img_area / np.sum(np.bitwise_and(self.image_mask == 255, cv2.fillPoly(mask, [approx_hull], 255) == 255)))

            M = cv2.moments(cnt)
            center_X = int(M["m10"] / M["m00"])
            center_Y = int(M["m01"] / M["m00"])
            centroid = (center_X, center_Y)

            rads = []
            for j in cnt.squeeze():
                rads.append(distance.euclidean(centroid, j))

            max_rad = max(rads)
            min_rad.append(np.round(min(rads) / max_rad, 6))
            mean_rad.append(np.round(np.mean(rads) / max_rad, 6))
            radial_deviation.append(np.round(np.mean(np.abs(np.array(rads) - max_rad)) / max_rad, 6))
            contour_roughness_1.append(np.round(np.mean(np.abs(rads - np.roll(rads, 1)) / max_rad), 6))
            contour_roughness_6.append(np.round(np.mean(np.abs(rads - np.roll(rads, len(cnt) // 6)) / max_rad), 6))

        return (
                self.len_not_procced_contours,
                len(self.cleaned_contours),
                np.round(bitwise_area_to_convex_hull, 6),
                np.round(total_area_to_convex_hull, 6),
                *self.statistics_for_contour_features(img_area_to_approx_hull),
                *self.statistics_for_contour_features(img_area_to_convex_hull),
                *self.statistics_for_contour_features(contour_area_to_convex_hull),
                *self.statistics_for_contour_features(min_rad),
                *self.statistics_for_contour_features(mean_rad),
                *self.statistics_for_contour_features(radial_deviation),
                *self.statistics_for_contour_features(contour_roughness_1),
                *self.statistics_for_contour_features(contour_roughness_6),
                convexity_changes
        )


    def get_texture_features(self):
        lesion = self.image.copy()
        lesion[self.image_mask != 255] = [0, 0, 0]
        lesion = cv2.cvtColor(lesion, cv2.COLOR_BGR2GRAY)

        total_features = []
        textures_haralick = mt.features.haralick(lesion)
        textures_haralick_mean = textures_haralick.mean(axis=0)
        total_features += list(textures_haralick_mean)

        num_points = 16
        radius = 3
        eps = 1e-7

        lbp = feature.local_binary_pattern(lesion, num_points, radius, method="uniform")
        (hist, _) = np.histogram(lbp.ravel(),
                                 bins=np.arange(0, num_points + 3),
                                 range=(0, num_points + 2))
        textures_lbp = hist.astype("float")
        textures_lbp /= (textures_lbp.sum() + eps)
        total_features += list(textures_lbp)

        hu = cv2.HuMoments(cv2.moments(self.image_mask))
        for i in range(0, 7):
            hu[i] = -1 * np.sign(hu[i]) * np.log10(np.abs(hu[i]))

        hu = hu.reshape((1, 7)).tolist()[0]
        total_features += list(hu)
        return total_features

    def get_all_features(self):
        features = []
        channel = [0, 1, 2]
        threshold = [10, 30, 50, 70, 90, 110, 130, 150, 170, 190, 210, 230, 250]
        mask_type = ['circle', 'ellipse', 'hull']
        coef = [0.05, 0.15, 0.25, 0.35, 0.45, 0.55, 0.65, 0.75, 0.85, 0.95, 1.0]

        features += self.get_texture_features()
        features += self.get_contour_features()
        for ch in channel:
            for th in threshold:
                for mt in mask_type:
                    for co in coef:
                        features.append(self.get_exact_area_feature(ch, th, mt, co))
        return features

    # ----- Ниже функции для итогового фича экстратора -----

    def get_total_contour_features(self):
        mask = np.zeros(self.mask_shape)
        mask = cv2.fillPoly(mask, [self.convex_hull], 255)
        mask_area = np.sum(mask == 255)
        bitwise_area_to_convex_hull = np.sum((np.bitwise_and(self.image_mask == 255, mask == 255))) / mask_area

        total_area_to_convex_hull = np.sum(self.image_mask == 255) / mask_area

        contour_area_to_convex_hull = []
        img_area_to_convex_hull = []
        img_area_to_approx_hull = []

        contour_roughness_1 = []
        contour_roughness_6 = []

        min_rad = []
        mean_rad = []

        radial_deviation = []

        convexity_changes = 0

        for cnt in self.cleaned_contours:
            mask = np.zeros(self.mask_shape)
            img_area = np.sum(np.bitwise_and(self.image_mask == 255, cv2.fillPoly(mask, [cnt], 255) == 255))
            cnt_area = cv2.contourArea(cnt)
            convex_hull_contour_area = cv2.contourArea(cv2.convexHull(cnt))

            contour_area_to_convex_hull.append(cnt_area / convex_hull_contour_area)
            img_area_to_convex_hull.append(img_area / convex_hull_contour_area)

            epsilon = 0.005 * cv2.arcLength(cnt, True)
            approx_hull = cv2.approxPolyDP(cnt, epsilon, True)

            prev_convexity = None
            for i in range(len(approx_hull)):
                p1 = approx_hull[i][0]
                p2 = approx_hull[(i + 1) % len(approx_hull)][0]
                p3 = approx_hull[(i + 2) % len(approx_hull)][0]
                cross_product = np.cross(p1 - p2, p3 - p2)

                # Определение выпуклости или вогнутости
                current_convexity = cross_product > 0

                if prev_convexity is not None and current_convexity != prev_convexity:
                    convexity_changes += 1

                prev_convexity = current_convexity

            img_area_to_approx_hull.append(
                img_area / np.sum(np.bitwise_and(self.image_mask == 255, cv2.fillPoly(mask, [approx_hull], 255) == 255)))

            M = cv2.moments(cnt)
            center_X = int(M["m10"] / M["m00"])
            center_Y = int(M["m01"] / M["m00"])
            centroid = (center_X, center_Y)

            rads = []
            for j in cnt.squeeze():
                rads.append(distance.euclidean(centroid, j))

            max_rad = max(rads)
            min_rad.append(np.round(min(rads) / max_rad, 6))
            mean_rad.append(np.round(np.mean(rads) / max_rad, 6))
            radial_deviation.append(np.round(np.mean(np.abs(np.array(rads) - max_rad)) / max_rad, 6))
            contour_roughness_1.append(np.round(np.mean(np.abs(rads - np.roll(rads, 1)) / max_rad), 6))
            contour_roughness_6.append(np.round(np.mean(np.abs(rads - np.roll(rads, len(cnt) // 6)) / max_rad), 6))

        return (
                self.len_not_procced_contours,  # lnpc
                np.round(bitwise_area_to_convex_hull, 6),  # batch
                np.round(total_area_to_convex_hull, 6),  #tatch
                np.max(img_area_to_approx_hull),  # max_iatah
                np.round(np.mean(img_area_to_approx_hull), 6),  # mean_iatah
                np.max(img_area_to_convex_hull),  # max_iatch
                np.round(np.mean(img_area_to_convex_hull), 6),  # mean_iatch
                np.min(contour_area_to_convex_hull),  # min_catch
                np.max(contour_area_to_convex_hull),  # max_catch
                np.min(contour_roughness_1),  # min_cr1
                np.max(contour_roughness_1),  # max_cr1
                np.min(contour_roughness_6),  # min_cr6
                np.max(contour_roughness_6),  # max_cr6
                np.round(np.mean(contour_roughness_6), 6),  # mean_cr6
                np.min(min_rad),  # min_mr
                np.max(min_rad),  # max_mr
                np.round(np.mean(min_rad), 6),  # mean_mr
                np.min(mean_rad),  #min_mnr
                np.min(radial_deviation),  #min_rd
                convexity_changes  #cc
        )

    def get_total_texture_features(self):
        lesion = self.image.copy()
        lesion[self.image_mask != 255] = [0, 0, 0]
        lesion = cv2.cvtColor(lesion, cv2.COLOR_BGR2GRAY)

        textures_haralick = mt.features.haralick(lesion)
        textures_haralick_mean = textures_haralick.mean(axis=0)

        num_points = 16
        radius = 3
        eps = 1e-7

        lbp = feature.local_binary_pattern(lesion, num_points, radius, method="uniform")
        (hist, _) = np.histogram(lbp.ravel(),
                                 bins=np.arange(0, num_points + 3),
                                 range=(0, num_points + 2))
        textures_lbp = hist.astype("float")
        textures_lbp /= (textures_lbp.sum() + eps)

        hu = cv2.HuMoments(cv2.moments(self.image_mask))
        for i in range(0, 7):
            hu[i] = -1 * np.sign(hu[i]) * np.log10(np.abs(hu[i]))

        hu = hu.reshape((1, 7)).tolist()[0]

        total_features = []
        for i in [2, 6, 11, 12]:
            total_features.append(textures_haralick_mean[i])
        for i in [0, 4, 5, 6, 7, 14, 15]:
            total_features.append(textures_lbp[i])
        for i in range(7):
            total_features.append(hu[i])
        return total_features

    def get_color_feature(self, img2process, colorspace, channel, statistic):
        if channel == 'full':
            ravel = img2process.ravel()
        else:
            if colorspace in ['hsv', 'yuv', 'rgb', 'xyz']:
                channel = colorspace.find(channel)
            else:
                channel = 'rgb'.find(channel)
            ravel = img2process[:, :, channel].ravel()
        distr = ravel[ravel > 0]
        if 'Q' in statistic:
            return np.quantile(distr, float('0.' + statistic[1:]))
        elif statistic == 'std':
            return np.std(distr)
        elif statistic == 'mean':
            return np.mean(distr)
        elif statistic == 'median':
            return np.median(distr)
        elif statistic == 'min':
            return np.min(distr)
        elif statistic == 'max':
            return np.max(distr)
        elif statistic == 'mode':
            return mode(distr).mode
        else:
            print('Error color feature extraction')
            return 0

    @staticmethod
    def preprocess_img_for_color_feature_extraction(image, params):
        if 'sobel_8u' in params['feature_name']:
            sobelx64f = cv2.Sobel(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), cv2.CV_64F, 1, 0, ksize=5)
            abs_sobel64f = np.absolute(sobelx64f)
            return np.uint8(abs_sobel64f)
        elif params['colorspace'] == 'hsv':
            return cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        elif params['colorspace'] == 'xyz':
            return cv2.cvtColor(image, cv2.COLOR_BGR2XYZ)
        elif params['colorspace'] == 'yuv':
            return cv2.cvtColor(image, cv2.COLOR_BGR2YUV)
        elif params['colorspace'] == 'laplacian':
            return cv2.Laplacian(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), cv2.CV_64F)
        elif params['colorspace'] == 'sobelx':
            return cv2.Sobel(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), cv2.CV_64F, 1, 0, ksize=5)
        elif params['colorspace'] == 'sobely':
            return cv2.Sobel(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), cv2.CV_64F, 0, 1, ksize=5)
        elif params['colorspace'] == 'sobelx2':
            return cv2.Sobel(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), cv2.CV_64F, 2, 0, ksize=5)
        elif params['colorspace'] == 'sobely2':
            return cv2.Sobel(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), cv2.CV_64F, 0, 2, ksize=5)
        elif params['colorspace'] == 'sobelx8u':
            return cv2.Sobel(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), cv2.CV_8U, 1, 0, ksize=5)
        elif params['colorspace'] == 'rgb' or params['colorspace'] == 'rbg':  # последствия опечатки при изначальном составлении датасета
            return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        else:
            print('Wrong colorspace')
            return -1

    def get_total_features(self, color_feats, area_feats):
        image, all_lesion_and_skin, lesion, skin = self.get_masks_for_color_feats()
        total_feats = []

        for feat in color_feats:
            if feat['image_type'] == 'ALAS':
                feat_image = all_lesion_and_skin.copy()
            elif feat['image_type'] == 'L':
                feat_image = lesion.copy()
            elif feat['image_type'] == 'S':
                feat_image = skin.copy()
            elif feat['image_type'] == 'I':
                feat_image = image.copy()
            else:
                print('Wrong feature image_type')
                return -1
            processed_image = self.preprocess_img_for_color_feature_extraction(feat_image, feat)
            total_feats.append(self.get_color_feature(processed_image, feat['colorspace'], feat['channel'], feat['statistic']))

        for feat in area_feats:
            total_feats.append(self.get_exact_area_feature(feat['channel'], int(feat['threshold']), feat['mask_type'], float(feat['coef'])))

        contour_features = self.get_total_contour_features()
        total_feats += list(contour_features)
        texture_features = self.get_total_texture_features()
        total_feats += list(texture_features)

        return total_feats

    def get_total_feature_names(self, color_feats, area_feats):
        total_feature_names = []
        for feat in color_feats:
            total_feature_names.append(feat['feature_name'])
        for feat in area_feats:
            total_feature_names.append(feat['feature_name'])
        total_feature_names += ['lnpc', 'batch', 'tatch', 'max_iatah', 'mean_iatah', 'max_iatch', 'mean_iatch',
                                'min_catch', 'max_catch', 'min_cr1', 'max_cr1', 'min_cr6', 'max_cr6', 'mean_cr6',
                                'min_mr', 'max_mr', 'mean_mr', 'min_mnr', 'min_rd', 'cc']
        total_feature_names += ['haralick3', 'haralick7', 'haralick12', 'haralick14', 'lbp1', 'lbp5', 'lbp6', 'lbp7',
                                'lbp8', 'lbp15', 'lbp16', 'hu1', 'hu2', 'hu3', 'hu4', 'hu5', 'hu6', 'hu7']
        return total_feature_names




