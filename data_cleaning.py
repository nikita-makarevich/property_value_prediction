import pandas as pd


class DataCleaning():

    def __init__(self, data: pd.DataFrame) -> None:
       self.data = data

    # Отфильтровываем строки
    def filter_outliers(self):
        self.data = self.data[self.data['id'] > 0]
        self.data = self.data[self.data['city_id'] > 0]
        self.data = self.data[self.data['district_id'] > 0]
        self.data = self.data[self.data['street_id'] > 0]
        self.data = self.data[(self.data['sold_price'] >= 500) & (self.data['sold_price'] <= 100000)]
        self.data = self.data[self.data['metro_station_id'].isin([0, -1])]
        # self.data = self.data[(self.data['flat_on_floor'] >= 0) & (self.data['flat_on_floor'] <= 10)]  # много нулей
        self.data = self.data[(self.data['floor_num'] >= 1) & (self.data['floor_num'] <= 30)]
        self.data = self.data[(self.data['floors_cnt'] >= 1) & (self.data['floors_cnt'] <= 30)]
        self.data = self.data[(self.data['rooms_cnt'] >= 1) & (self.data['rooms_cnt'] <= 6)]
        self.data = self.data[(self.data['bedrooms_cnt'] >= 0) & (self.data['bedrooms_cnt'] <= 4)]  # много нулей
        self.data = self.data[(self.data['building_year'] >= 1900) & (self.data['building_year'] <= 2024)]
        self.data = self.data[(self.data['area_total'] >= 12) & (self.data['area_total'] <= 150)]
        self.data = self.data[
            (self.data['area_live'] >= 8) & (self.data['area_live'] <= 90) | (self.data['area_live'] == 0)
        ]  # много нулей
        self.data = self.data[
            (self.data['area_kitchen'] >= 5) & (self.data['area_kitchen'] <= 30) | (self.data['area_kitchen'] == 0)
        ]  # много нулей
        # self.data['area_balcony'] = pd.to_numeric(self.data['area_balcony'], errors='coerce')
        # self.data['area_balcony'] = self.data['area_balcony'].astype(float)
        # self.data = self.data[
        # (self.data['area_balcony'] <= 15) & (self.data['area_balcony'] >= 1) | (self.data['area_balcony'] == 0)
        # ]
        # self.data = self.data[(self.data['builder_id'] >= 0)] # много нулей
        self.data = self.data[(self.data['levels_count'] >= 1) & (self.data['levels_count'] <= 3)]
        self.data = self.data[
            (
                    (self.data['bathrooms_cnt'] >= 0) & (self.data['bathrooms_cnt'] <= 4)
            ) | (pd.isna(self.data['bathrooms_cnt']))
            ] # заменить нули на единицу
        self.data = self.data[(self.data['series_id'] > 0)]
        self.data = self.data[(self.data['wall_id'] > 0)]
        # self.data = self.data[self.data['loggia'].isin([0, 1])]
        self.data = self.data[
            (
                    self.data['ceiling_height'] >= 2.4) & (self.data['ceiling_height'] <= 5)
            | (self.data['ceiling_height'] == 0)
        ]  # много нулей, порпобовать заменить на 2.5

    # Функция для обработки значений в столбце
    def split_komunal_cost(self, el):
        try:
            el = float(el)
        except ValueError:
            pass
        try:
            num1, num2 = map(float, el.split('-'))
        except ValueError:
            return 0     
           
        except AttributeError:
            return el        
        
        return (num1 + num2) / 2
    
    def split_territory(self):
        # Используем get_dummies для разбиения territory на отдельные бинарные столбцы
        territory_dummies = self.data['territory'].str.get_dummies(sep=',')
        self.data = pd.concat([self.data, territory_dummies], axis=1)
        self.data.drop(columns=['territory'], inplace=True)

    def filter_plate(self, el):

        if el in ['electric', 'no_plate', 'gas', 'convective']:
            return el
        return 'unknown'

    def filter_bathrooms_cnt(self, el):

        if 1 < el < 4:
            return el
        return 1

    def transform_for_1d(self, column_name, n=50):
        
        freq = self.data[column_name].value_counts()
        rare_categories = freq.index[n:]
        self.data[column_name] = self.data[column_name].apply(lambda x: x if x in freq.index[:n] else 'rare_'+column_name)

    def main(self):
        self.data = self.data.drop([
            'price',
            'status',
            'area_balcony',
            'komunal_cost',
            'closed_yard',
            'flat_on_floor',
            'loggia',
            'builder_id'
        ], axis=1)
        self.filter_outliers()
        # self.data['komunal_cost'] = self.data['komunal_cost'].apply(self.split_komunal_cost)
        self.data['plate'] = self.data['plate'].apply(self.filter_plate)
        self.data['bathrooms_cnt'] = self.data['bathrooms_cnt'].apply(self.filter_bathrooms_cnt)

        col_id = ['city_id', 'district_id', 'street_id', 'series_id', 'wall_id']
        self.data[col_id] = self.data[col_id].astype(str)       
        for el in col_id:
            self.transform_for_1d(el)

        self.split_territory()

        self.data = self.data.drop_duplicates()
        clean_data = self.data

        return clean_data
        
