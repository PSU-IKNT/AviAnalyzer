#моудль для конвертации всех json файлов в csv с добавлением униакльных кодов и хэдера
#для работы поместите их в папку
import os
import json
import csv

data_list = []
for filename in os.listdir("data_to_analyze"):
    if filename.endswith(".json"):
        with open(os.path.join("fact", filename), "r") as file:
            flight_data = json.load(file)

            flight_id = os.path.splitext(filename)[0]

            airline_iata_code = flight_data["airline_iata_code"]
            flight = flight_data["flight"]
            departure_airport_code = flight_data["departure_airport"]
            arrival_airport_code = flight_data["arrival_airport"]
            plan_departure = flight_data["plan_departure"]
            plan_arrival = flight_data["plan_arrival"]
            fact_departure = flight_data["fact_departure"]
            fact_arrival = flight_data["fact_arrival"]

            data_list.append([flight_id, airline_iata_code, flight, departure_airport_code, arrival_airport_code, plan_departure, plan_arrival, fact_departure, fact_arrival])

with open("flight_data.csv", "w", newline="") as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(["flight_id", "airline_iata_code", "flight", "departure_airport_code", "arrival_airport_code", "plan_departure", "plan_arrival", "fact_departure", "fact_arrival"])
    writer.writerows(data_list)

print("CSV файл успешно создан!")