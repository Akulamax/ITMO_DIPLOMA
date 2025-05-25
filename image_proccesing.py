import cv2
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial import distance


class ImageProcessing:

    def __init__(self, image_mask_path: str):
        self.len_not_procced_contours = None
        self.image_mask = cv2.imread(image_mask_path, cv2.IMREAD_UNCHANGED)
        self.image_mask_to_feature = self.image_mask.copy()
        self.image = cv2.imread(image_mask_path[:-9] + '.jpg', 1)
        self.mask_shape = self.image_mask.shape
        self.cleaned_contours = self.get_cleaned_contours()
        self.convex_hull = self.get_convex_hull()
        self.centroid = self.get_centroid()

    def get_circle_radius(self) -> int:
        radius = 0
        for i in self.convex_hull:
            d = distance.euclidean(self.centroid, i)
            if d > radius:
                radius = d
        return int(radius)

    def get_ellipse_params(self) -> tuple[int, int, float, float, float]:
        # Функция для вычисления расстояния от точек полигона до эллипса
        def distance_to_ellipse(params, points):
            x0, y0, a, b, theta = params
            cos_theta = np.cos(theta)
            sin_theta = np.sin(theta)
            distances = []
            for x, y in points:
                dx = x - x0
                dy = y - y0
                x_rot = cos_theta * dx + sin_theta * dy
                y_rot = -sin_theta * dx + cos_theta * dy
                distance = (x_rot / a) ** 2 + (y_rot / b) ** 2 - 1
                distances.append(distance)
            return np.array(distances)

        # начальные предположения для параметров эллипса
        initial_guess = [self.centroid[0], self.centroid[1], 1, 1, 0]

        # оптимизация для нахождения параметров эллипса
        result = least_squares(distance_to_ellipse, initial_guess, args=(self.convex_hull,))
        x0, y0, a, b, theta = result.x
        return int(x0), int(y0), b, a, theta

    def get_convex_hull(self) -> np.array:
        length = len(self.cleaned_contours)
        # concatinate poits form all shapes into one array
        cont = np.vstack([self.cleaned_contours[i] for i in range(length)])
        hull = cv2.convexHull(cont).squeeze()
        # cv2.drawContours(image,uni_hull,-1,255,2);
        # cv2.fillPoly(new_mask, [uni_hull[0]], 255)
        return hull

    def get_centroid(self) -> tuple[int, int]:
        m = cv2.moments(self.convex_hull)
        center_x = int(m["m10"] / m["m00"])
        center_y = int(m["m01"] / m["m00"])
        centroid = (center_x, center_y)
        return centroid

    def get_cleaned_contours(self):
        # оставить только маску поражения кожи (может поменять на threshold)
        image_mask = self.image_mask[self.image_mask < 200] = 0
        self.image_mask[self.image_mask >= 200] = 255

        # сделаем морфологическую операцию открытия,
        # чтобы отделить мелкие вкрапления и волосы от основного поражения
        kernel = np.ones((2, 2), np.uint8)
        img = cv2.morphologyEx(self.image_mask, cv2.MORPH_OPEN, kernel, iterations=7)

        # построим все контуры
        contours, hierarchy = cv2.findContours(img, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        if len(contours) == 0:
            img = cv2.morphologyEx(self.image_mask, cv2.MORPH_OPEN, kernel, iterations=3)
            contours, hierarchy = cv2.findContours(img, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

            if len(contours) == 0:
                img = self.image_mask

        self.len_not_procced_contours = len(contours)

        # отсортируем контуры по площади, от большего к меньшему.
        sorted_contours = sorted(contours, key=cv2.contourArea, reverse=True)

        # оставим только самые большие области поражения
        largest_contour_area = cv2.contourArea(sorted_contours[0])
        cleaned_contours = [sorted_contours[0]]
        for contour in sorted_contours[1:]:
            if cv2.contourArea(contour) * 5 >= largest_contour_area:
                cleaned_contours.append(contour)
            else:
                break
        tmp_mask = np.zeros(self.mask_shape)
        for i in cleaned_contours:
            tmp_mask = cv2.fillPoly(tmp_mask.astype('uint8'), [i], 255);

        cleaned_contours, _ = cv2.findContours(tmp_mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

        return cleaned_contours

    def get_main_parameters_for_masks(self):
        parameters = {'circle_radius': self.get_circle_radius(),
                      'ellipse_params': self.get_ellipse_params(),
                      'convex_hull': self.convex_hull,
                      'centroid': self.centroid}
        return parameters

    def get_masks_for_color_feats(self):
        image = self.image.copy()
        mask = self.image_mask_to_feature.copy()
        image[mask < 15] = [0, 0, 0]
        lesion = mask.copy()
        lesion[np.logical_and(0 < mask, mask < 170)] = 0
        skin = mask.copy()
        skin[mask > 0] = 0
        skin[np.logical_and(10 < mask, mask < 170)] = 255

        contours, hierarchy = cv2.findContours(image=lesion, mode=cv2.RETR_TREE, method=cv2.CHAIN_APPROX_NONE)
        sorted_cnt = sorted(contours, key=cv2.contourArea, reverse=True)
        if len(sorted_cnt) == 0:
            print('Не удалось выделить контур поражения')
            return image, image, image, image, image
        cnt = sorted_cnt[0]
        if sorted_cnt[0][0][0][0] == 0 and sorted_cnt[0][0][0][1] == 0:
            cnt = sorted_cnt[1]

        if len(cnt) != 0:
            x, y, w, h = cv2.boundingRect(cnt)
            all_lesion_and_skin = image[y:y + h, x:x + w]
        else:
            print('Не удалось выделить изображение кожи и всего поражения на изображении')
            all_lesion_and_skin = image

        lesion = cv2.bitwise_and(image, image, mask=lesion)
        skin = cv2.bitwise_and(image, image, mask=skin)

        return image, all_lesion_and_skin, lesion, skin