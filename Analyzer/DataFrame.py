import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules
import glob
import logging


class FlightAnalysis:
    def __init__(self, path):
        self.path = path
        self.df = None
        self.df_airlines = None
        self.rules = None
        logging.basicConfig(filename='Analyze.log', level=logging.INFO)

    def load_data(self):
        data = pd.read_csv(self.path, sep=',')
        self.df = pd.DataFrame(data)

    def preprocess_data(self):
        print(self.df.head())
        self.df['plan_departure'] = pd.to_datetime(self.df['plan_departure'])
        self.df['plan_arrival'] = pd.to_datetime(self.df['plan_arrival'])
        self.df['fact_departure'] = pd.to_datetime(self.df['fact_departure'])
        self.df['fact_arrival'] = pd.to_datetime(self.df['fact_arrival'])
        self.df['delay'] = (self.df['fact_arrival'] - self.df['plan_arrival']).dt.total_seconds() / 60
        self.df['X1'] = (self.df['delay'] >= 0) & (self.df['delay'] <= 15)
        self.df['X2'] = (self.df['delay'] > 15) & (self.df['delay'] <= 60)
        self.df['X3'] = (self.df['delay'] > 60) & (self.df['delay'] <= 180)
        self.df['X4'] = self.df['delay'] > 180

    def create_airline_df(self):
        unique_airlines = self.df['airline_iata_code'].unique()
        self.df_airlines = pd.DataFrame(False, index=self.df.index, columns=unique_airlines)
        for index, row in self.df.iterrows():
            airline = row['airline_iata_code']
            self.df_airlines.loc[index, airline] = True
        self.df_airlines.insert(0, 'flight_id', self.df['flight_id'])
        self.df_airlines['X1'] = self.df['X1']
        self.df_airlines['X2'] = self.df['X2']
        self.df_airlines['X3'] = self.df['X3']
        self.df_airlines['X4'] = self.df['X4']

    def save_airline_df(self, filename):
        self.df_airlines.to_csv(filename, index=False)

    def load_airline_df(self, filename):
        self.df_airlines = pd.read_csv(filename)
        self.df_airlines = self.df_airlines.drop('flight_id', axis=1)

    def compute_rules(self):
        frequent_itemsets = apriori(self.df_airlines, min_support=0.001, use_colnames=True)
        self.rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=0.001)
        self.rules['lift'] = self.rules['support'] / (
                self.rules['antecedent support'] * self.rules['consequent support'])
        self.rules = self.rules.sort_values(by='conviction', ascending=False)
        self.rules = self.rules.round(5)

    def save_rules(self, filename):
        self.rules.to_json(filename, orient='records', lines=True)


if __name__ == "__main__":
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    handler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    flight_analysis = FlightAnalysis('Analyzer/flights_with_id.csv')
    try:
        flight_analysis.load_data()
        print("DONE1")
    except Exception as e:
        logging.error(f'Ошибка возникла во время загрузки данных: {e}')
    flight_analysis.preprocess_data()
    print("DONE2")
    flight_analysis.create_airline_df()
    print("DONE3")
    flight_analysis.save_airline_df('airlines.csv')
    print("DONE4")
    flight_analysis.load_airline_df('airlines.csv')
    print("DONE5")
    flight_analysis.compute_rules()
    flight_analysis.save_rules('association_rules.json')
    print("DONE6")