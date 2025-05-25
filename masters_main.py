import time
from feature_extraction import FeatureExtraction
import pandas as pd
import os
from tqdm import tqdm


def get_dataframe_features():
    columns = []
    contour_columns = ['lnpc', 'lcc', 'batch', 'tatch',
                       'min_iatah', 'mean_iatah', 'max_iatah',
                       'min_iatch', 'mean_iatch', 'max_iatch',
                       'min_catch', 'mean_catch', 'max_catch',
                       'min_mr', 'mean_mr', 'max_mr',
                       'min_mnr', 'mean_mnr', 'max_mnr',
                       'min_rd', 'mean_rd', 'max_rd',
                       'min_cr1', 'mean_cr1', 'max_cr1',
                       'min_cr6', 'mean_cr6', 'max_cr6', 'cc']

    texture_columns = ['haralick' + f'{i}' for i in range(1, 14)]
    texture_columns += ['lbp' + f'{i}' for i in range(1, 19)]
    texture_columns += ["hu1", "hu2", "hu3", "hu4", "hu5", "hu6", "hu7"]

    channel = [0, 1, 2]
    threshold = [10, 30, 50, 70, 90, 110, 130, 150, 170, 190, 210, 230, 250]
    mask_type = ['circle', 'ellipse', 'hull']
    coef = [0.05, 0.15, 0.25, 0.35, 0.45, 0.55, 0.65, 0.75, 0.85, 0.95, 1.0]

    area_columns = []
    for ch in channel:
        for th in threshold:
            for mt in mask_type:
                for co in coef:
                    area_columns.append(f"area_{ch}_{th}_{mt}_{co}")

    columns += texture_columns
    columns += contour_columns
    columns += area_columns
    columns += ['image_name', 'target']
    dataframe = pd.DataFrame(columns=columns)
    return dataframe


def main():
    tmp = pd.read_csv('/Users/mikhailledovskikh/Учеба/skin lesions/total_skin_color_dataset.csv')
    tmp2 = pd.read_csv('/Users/mikhailledovskikh/Учеба/диплом/texture_contour_area_features_dataset_clean_masked_AK_BKL_DF_SCC_VASC_1_first_cut.csv')
    processed_yet = set(tmp2['image_name'])
    image_classes = {}
    for img_name, img_class in tmp[['image_name', 'target']].values:
        if img_name not in processed_yet:
            image_classes[img_name] = img_class

    dataframe = get_dataframe_features()
    index = 1
    for dir in os.listdir('/Users/mikhailledovskikh/Учеба/диплом'):
        if dir != '.DS_Store' and dir[:5] == 'clean':
            for image_mask in tqdm(os.listdir('/Users/mikhailledovskikh/Учеба/диплом/' + dir)):
                image_name = image_mask[:-9]
                if image_name in image_classes:
                    image_class = image_classes[image_name]
                    try:
                        fe = FeatureExtraction(
                            '/Users/mikhailledovskikh/Учеба/skin lesions/' + image_class + '/' + image_name + '.jpg',
                            '/Users/mikhailledovskikh/Учеба/диплом/' + dir + '/' + image_mask)
                        dataframe.loc[len(dataframe.index)] = fe.get_all_features() + [image_name, image_class]
                    except Exception as E:
                        print(f'some problem with {image_name}: ', E)
                    if index % 1000 == 0:
                        dataframe.to_csv(
                            f'/Users/mikhailledovskikh/Учеба/диплом/texture_contour_area_features_dataset_{dir}_{index}.csv')
                    index += 1
    dataframe.to_csv('/Users/mikhailledovskikh/Учеба/диплом/texture_contour_area_features_dataset_last.csv')


if __name__ == '__main__':
    main()
