import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules
import avi_constants


class FlightAnalysis:
    def __init__(self, path):
        self.path = path
        self.df = None
        self.df_airlines = None
        self.rules = None

    def load_data(self):
        data = pd.read_csv(self.path, sep=',')
        self.df = pd.DataFrame(data)

    def preprocess_data(self):
        self.df['plan_departure'] = pd.to_datetime(self.df['plan_departure'])
        self.df['plan_arrival'] = pd.to_datetime(self.df['plan_arrival'])
        self.df['fact_departure'] = pd.to_datetime(self.df['fact_departure'])
        self.df['fact_arrival'] = pd.to_datetime(self.df['fact_arrival'])
        self.df['delay'] = (self.df['fact_arrival'] - self.df['plan_arrival']).dt.total_seconds() / 60
        self.df['X1'] = (self.df['delay'] >= 0) & (self.df['delay'] <= 15)
        self.df['X2'] = (self.df['delay'] > 15) & (self.df['delay'] <= 60)
        self.df['X3'] = (self.df['delay'] > 60) & (self.df['delay'] <= 180)
        self.df['X4'] = self.df['delay'] > 180
        for i in range(0, 4):
            self.df[f'{avi_constants.months[i]}'] = (self.df["plan_departure"].dt.month.apply(avi_constants.get_season)
                                                     == avi_constants.months[i])
        for i in range(0, 7):
            self.df[avi_constants.day_of_weeks[i]] = self.df['fact_departure'].dt.dayofweek == i
        # for i in range(0, 24):
        #    self.df[f'{avi_constants.hours[i]}'] = (
        #                self.df["plan_departure"].dt.hour.apply(avi_constants.get_hour_range) == avi_constants.hours[i])
        self.df['00:00-05:59'] = (self.df['plan_departure'].dt.hour >= 0) & (self.df['plan_departure'].dt.hour < 6)
        self.df['06:00-11:59'] = (self.df['plan_departure'].dt.hour >= 6) & (self.df['plan_departure'].dt.hour < 12)
        self.df['12:00-17:59'] = (self.df['plan_departure'].dt.hour >= 12) & (self.df['plan_departure'].dt.hour < 18)
        self.df['18:00-23:59'] = (self.df['plan_departure'].dt.hour >= 18)
        #self.df.to_csv("test" + "_V1", index=False)

    def create_airline_df(self):
        unique_airlines = self.df['airline_iata_code'].unique()
        self.df_airlines = pd.DataFrame(False, index=self.df.index, columns=unique_airlines)
        for index, row in self.df.iterrows():
            airline = row['airline_iata_code']
            self.df_airlines.loc[index, airline] = True
        self.df_airlines.insert(0, 'flight_id', self.df['flight_id'])
        for i in range(0, 4):
            self.df_airlines[f'{avi_constants.months[i]}'] = self.df[f'{avi_constants.months[i]}']
        for i in range(0, 4):
            self.df_airlines[f'{avi_constants.day_parts[i]}'] = self.df[f'{avi_constants.day_parts[i]}']
        for i in range(0, 4):
            self.df_airlines[f'{avi_constants.delays[i]}'] = self.df[f'X{i + 1}']
        for i in range(0, 7):
            self.df_airlines[f'{avi_constants.day_of_weeks[i]}'] = self.df[f'{avi_constants.day_of_weeks[i]}']
        # for i in range(0, 24):
        #    self.df_airlines[f'{avi_constants.hours[i]}'] = self.df[f'{avi_constants.hours[i]}']

    def save_airline_df(self, filename):
        # self.df.to_csv(filename + "_V1", index=False)
        self.df_airlines.to_csv(filename, index=False)

    def load_airline_df(self, filename):
        self.df_airlines = pd.read_csv(filename)

    def compute_rules(self):
        self.df_airlines = self.df_airlines.drop('flight_id', axis=1)
        frequent_itemsets = apriori(self.df_airlines, min_support=0.0001, use_colnames=True)
        self.rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=0.1)
        self.rules = self.rules.sort_values(by='leverage', ascending=False)
        self.rules = self.rules.round(5)

    def save_raw_rules(self, filename):
        self.rules.to_json(filename, orient='records', lines=True, force_ascii=False)
        filename.encode().decode()

    # Русские названия ассоциативных параметров и т.д. + в процентаже вывести
    def save_business_rules(self, filename):
        self.rules.rename(columns=avi_constants.business_column_names, inplace=True)
        self.rules.to_json(filename, orient='records', lines=True, force_ascii=False)


if __name__ == "__main__":
    flight_analysis = FlightAnalysis('.\\flights_with_id.csv')
    try:
        flight_analysis.load_data()
    except Exception as e:
        print(f'Ошибка возникла во время загрузки данных: {e}')
    flight_analysis.preprocess_data()
    flight_analysis.create_airline_df()
    flight_analysis.save_airline_df('airlines.csv')
    # flight_analysis.load_airline_df('airlines.csv')
    flight_analysis.compute_rules()
    flight_analysis.save_raw_rules('association_rules.json')
    flight_analysis.save_business_rules('association_rules_BS.json')
