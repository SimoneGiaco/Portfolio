import pandas as pd
import csv
from datetime import datetime
from data_entry import get_date, get_amount, get_category, get_description
import matplotlib.pyplot as plt


#We define a class to create and update a CSV file containing data about financial transactions. It has three class methods to initialize the file, read it and update it.
class CSV():
# Class variables providing the name of the csv file, the names of its columns and the chosen format for the dates.
    CSV_FILE="finance_data.csv"
    COLUMNS=['date','amount','category','description']
    FORMAT="%d-%m-%Y"

# Function which reads the file CSV_FILE if it exists or creates it
    @classmethod
    def initialize_csv(cls):
        try:
            pd.read_csv(cls.CSV_FILE)
        except:
            FileNotFoundError
            df=pd.DataFrame(columns=cls.COLUMNS)
            df.to_csv(cls.CSV_FILE, index=False)

# Function which adds a new line to the file CSV_FILE
    @classmethod
    def add_entry(cls,date,amount,category,description):
        new_entry = {
            'date': date,
            'amount': amount,
            'category': category,
            'description': description,
        }
        with open(cls.CSV_FILE, 'a',newline="") as csvfile:
            writer=csv.DictWriter(csvfile, fieldnames=cls.COLUMNS)
            writer.writerow(new_entry)
        print("Entry added successfully!")

# Function which retrieves all the transactions in the period [start_date, end_date]
    @classmethod
    def get_transactions(cls, start_date, end_date):
        start_date= datetime.strptime(start_date, CSV.FORMAT)
        end_date= datetime.strptime(end_date, CSV.FORMAT)
        df=pd.read_csv(cls.CSV_FILE)
        df['date']=pd.to_datetime(df['date'], format=CSV.FORMAT)

        # We define a mask to select only transactions in the desired date range.
        mask= (df['date']>=start_date) & (df['date']<=end_date)
        filtered_df= df.loc[mask]

        if filtered_df.empty:
            print("No transactions found in the given date range.")
        else:
            print(
                f"Transactions from {start_date.strftime(CSV.FORMAT)} to {end_date.strftime(CSV.FORMAT)}"
            )
            print(
                filtered_df.to_string(index= False, formatters={'date': lambda x: x.strftime(CSV.FORMAT)})
            )

            total_income=filtered_df[filtered_df['category']=='Income']['amount'].sum()
            total_expenses=filtered_df[filtered_df['category']=='Expense']['amount'].sum()
            print("\nSummary:")
            print(f"Total Income: {total_income:.2f}")
            print(f"Total Expenses: {total_expenses:.2f}")
            print(f"Total Savings: {(total_income - total_expenses):.2f}")
        return filtered_df

# Function which adds a row to the csv file. It exploits the class method add_entry and the helper functions defined in the file data_entry.py
def add():
    CSV.initialize_csv()
    date= get_date(
        "Enter the date in the format 'dd-mm-yyyy' or press enter for today's date: ",
        allow_default=True,
    )
    amount= get_amount()
    category= get_category()
    description= get_description()
    CSV.add_entry(date, amount, category, description)

# Function which plots the daily transactions (both incomes and expenses) using the Matplotlib library. 
def plot_transactions(df):
    #We first sort the values by date for readability of the plot
    df.sort_values('date', inplace=True)
    df.set_index('date', inplace=True)
     
    # We combine all transactions of the same category (income vs expenses) occurred in the same day
    income_df= df[df['category']=='Income'].resample('D').sum().reindex(df.index, fill_value=0)
    expense_df= df[df['category']=='Expense'].resample('D').sum().reindex(df.index, fill_value=0)

    plt.figure(figsize=(10,5))
    plt.plot(income_df.index, income_df['amount'], label='Income', color='g')
    plt.plot(expense_df.index, expense_df['amount'], label='Expense', color='r')
    plt.xlabel("Date")
    plt.ylabel("Amount")
    plt.title("Income and Expenses over time")
    plt.legend()
    plt.grid(True)
    plt.show()

# Main function which allows the user to input the chosen operation to perform
def main():
    while True:
        print("\n1. Add a new transaction")
        print("2. View transactions and summary in a given date range")
        print("3. Exit")
        choice= input("Enter your choice (1-3): ")

        if choice=="1":
            add()
        elif choice=="2":
            start_date= get_date("Enter the start date (dd-mm-yyyy): ")
            end_date= get_date("Enter the end date (dd-mm-yyyy): ")
            df= CSV.get_transactions(start_date, end_date)
            if input("Do you want to see a plot of the transactions? (y/n) ").lower()=="y":
                plot_transactions(df)
        elif choice=="3":
            print("Exiting...")
            break
        else:
            print("Invalid choice. Enter 1, 2 or 3.")


if __name__=="__main__":
    main()