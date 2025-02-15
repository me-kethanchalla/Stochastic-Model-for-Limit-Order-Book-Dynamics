import pandas as pd
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import poisson



def process_data(message_file_path):
    
    df_message = pd.read_csv(message_file_path)
    df_message["time_Bin"] = df_message["time"].astype(int)
    
    return df_message
    



def aggregate_order_counts(df_message, order_type1, order_type2, direction1=0, direction2=0):
   
    df_filtered1 = df_message[df_message["type"] == order_type1]
    if direction1 is not 0:
        df_filtered1 = df_filtered1[df_filtered1["dir"] == direction1]
        
    df_filtered2 = df_message[df_message["type"] == order_type2]
    if direction2 is not 0:
        df_filtered2 = df_filtered2[df_filtered2["dir"] == direction2]

    # merge 
    df_filtered = pd.concat([df_filtered1, df_filtered2])

    # counts per second
    order_counts = df_filtered.pivot_table(index="time_Bin", columns="type", aggfunc="size", fill_value=0)

    # calculate rates of both orders.
    if order_type1 in order_counts.columns and order_type2 in order_counts.columns:
        empirical_correlation = order_counts[order_type1].corr(order_counts[order_type2])
        lambda_1 = order_counts[order_type1].mean()
        lambda_2 = order_counts[order_type2].mean()
    else:
        empirical_correlation = None
        lambda_1, lambda_2 = None, None
    
    return order_counts, empirical_correlation, lambda_1, lambda_2





# GAUSSIAN COPULA


def simulate_orders(empirical_correlation, lambda_1, lambda_2, size=23400):
    
    
    empirical_correlation = np.clip(empirical_correlation, -0.99, 0.99)  # Ensure valid range
    
    mean = [0, 0]  # Standard normal mean
    cov_matrix = [[1, empirical_correlation], [empirical_correlation, 1]]  # Valid correlation matrix


    Z = np.random.multivariate_normal(mean, cov_matrix, size)
 
    # Transform to uniform distribution
    U1 = stats.norm.cdf(Z[:, 0])
    U2 = stats.norm.cdf(Z[:, 1])

    # Transform to Poisson-distributed counts
    X1 = stats.poisson.ppf(U1, lambda_1).astype(int)
    X2 = stats.poisson.ppf(U2, lambda_2).astype(int)

    # Compute simulated correlation
    simulated_corr = np.corrcoef(X1, X2)[0, 1]

    # Create DataFrame for simulated data
    df_simulated = pd.DataFrame({"Type 1": X1, "Type 2": X2})

    return df_simulated, simulated_corr





# TRIVARAITE POISSON COPULA


# def simulate_orders(empirical_correlation, lambda_1, lambda_2, size=23400):
  
#     empirical_correlation = np.clip(empirical_correlation, 0, 0.99)  

#     # Compute the shared Poisson intensity λ shared based on empirical correlation
#     lambda_shared = min(empirical_correlation * np.sqrt(lambda_1 * lambda_2), min(lambda_1, lambda_2))

#     # Ensure all Poisson parameters remain valid (non-negative)
#     lambda_1_indep = max(lambda_1 - lambda_shared, 0)
#     lambda_2_indep = max(lambda_2 - lambda_shared, 0)

#     # Generate independent Poisson-distributed counts
#     X1_independent = poisson.rvs(lambda_1_indep, size=size)
#     X2_independent = poisson.rvs(lambda_2_indep, size=size)


#     S_shared = poisson.rvs(lambda_shared, size=size)

#     X1 = X1_independent + S_shared
#     X2 = X2_independent + S_shared

#     simulated_corr = np.corrcoef(X1, X2)[0, 1]
#     df_simulated = pd.DataFrame({"Type 1": X1, "Type 2": X2})

#     return df_simulated, simulated_corr







def visualize_data(df_simulated, order_counts, order_type1, order_type2, empirical_correlation, simulated_corr):
    
    plt.figure(figsize=(16, 10))


    # Bar Chart of Order Counts Over Time (First 100 Intervals)
    plt.subplot(2, 2, 1)
    plt.bar(range(100), df_simulated["Type 1"][:100], color='blue', alpha=0.7, label="Type 1")
    plt.bar(range(100), df_simulated["Type 2"][:100], color='red', alpha=0.7, label="Type 2")
    plt.title("Bar Chart: Type 1 vs Type 2 (First 100 Intervals)")
    plt.xlabel("Time Interval")
    plt.ylabel("Order Count")
    plt.legend()

    # Full-Time Order Flow Comparison (Original vs Simulated)
    plt.subplot(2, 2, 2)
    plt.bar(df_simulated.index, df_simulated["Type 1"], color='blue', alpha=0.7, label="Type 1")
    plt.bar(df_simulated.index, df_simulated["Type 2"], color='red', alpha=0.7, label="Type 2")
    plt.title("Full-Time Type 1 vs Type 2")
    plt.xlabel("Time Interval")
    plt.ylabel("Order Count")
    plt.legend()

    #  Moving Average to Smooth Trends
    window_size = 100
    df_simulated["Moving Avg Type 1"] = df_simulated["Type 1"].rolling(window=window_size).mean()
    df_simulated["Moving Avg Type 2"] = df_simulated["Type 2"].rolling(window=window_size).mean()
    plt.subplot(2, 2, 3)
    plt.plot(df_simulated.index, df_simulated["Moving Avg Type 1"], label="Moving Avg Type 1", color='blue')
    plt.plot(df_simulated.index, df_simulated["Moving Avg Type 2"], label="Moving Avg Type 2", color='red')
    plt.title(f"Moving Average for Order Counts (Window Size = {window_size})")
    plt.xlabel("Time")
    plt.ylabel("Smoothed Order Count")
    plt.legend()

    #  Histogram of Simulated Poisson Counts
    plt.figure(figsize=(12, 6))
    sns.histplot(df_simulated["Type 1"], kde=True, color='blue', label='Type 1')
    sns.histplot(df_simulated["Type 2"], kde=True, color='red', label='Type 2', alpha=0.6)
    plt.title("Histogram of Simulated Poisson Counts")
    plt.xlabel("Count")
    plt.ylabel("Frequency")
    plt.legend()

    plt.tight_layout()
    plt.show()

    #  Original vs Simulated over time
    plt.figure(figsize=(12, 6))
    if order_type1 in order_counts.columns and order_type2 in order_counts.columns:
        plt.plot(order_counts.index, order_counts[order_type1], label="Original Type 1", color='blue')
        plt.plot(order_counts.index, order_counts[order_type2], label="Original Type 2", color='red', alpha=0.7)
    plt.plot(df_simulated.index, df_simulated["Type 1"], linestyle='dashed', color='blue', label="Type 1")
    plt.plot(df_simulated.index, df_simulated["Type 2"], linestyle='dashed', color='red', alpha=0.7, label="Type 2")
    plt.legend()
    plt.title("Original vs Simulated Order ")
    plt.xlabel("Time")
    plt.ylabel("Order Count")
    plt.show()

    #  Joint Distribution of Simulated vs Original Data 
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    sns.histplot(x=df_simulated["Type 1"], y=df_simulated["Type 2"], bins=30, cmap='coolwarm', cbar=True)
    plt.xlabel("Type 1")
    plt.ylabel("Type 2")
    plt.title("Joint Distribution: Simulated Type 1 & Type 2")

    plt.subplot(1, 2, 2)
    if order_type1 in order_counts.columns and order_type2 in order_counts.columns:
        sns.histplot(x=order_counts[order_type1], y=order_counts[order_type2], bins=30, cmap='coolwarm', cbar=True)
    plt.xlabel("Original Type 1")
    plt.ylabel("Original Type 2")
    plt.title("Joint Distribution: Original Type 1 & Type 2")

    plt.tight_layout()
    plt.show()







def get_order_type(order_num):
    
    while True:
        print(f"""Order Type {order_num}:
        1: Submission of a new limit order
        2: Cancellation (Partial deletion of a limit order)
        3: Deletion (Total deletion of a limit order)
        4: Execution of a visible limit order
        5: Execution of a hidden limit order""")
        try:
            order_type = int(input(f"Enter Order Type {order_num} (1-5): "))
            if order_type in {1, 2, 3, 4, 5}:
                return order_type
            else:
                print("Invalid input! Please enter a valid order type (1, 2, 3, 4, or 5).")
        except ValueError:
            print("Invalid input! Please enter a number.")
            
            
            
            

def get_direction(order_num):
    
    while True:
        print(f"""Direction for Order {order_num}:
        -1: Sell 
         1: Buy 
         0: Both """)
        try:
            direction = int(input(f"Enter Direction for Order {order_num} (-1, 0, or 1): "))
            if direction in {-1, 0, 1}:
                return direction
            else:
                print("Invalid input! Please enter -1 (Sell), 1 (Buy), or 0 (Both).")
        except ValueError:
            print("Invalid input! Please enter a number.")






if __name__ == "__main__":

    message_file_path = "AAPL_2012-06-21_34200000_57600000_message_1.csv"  
    df_message = process_data(message_file_path)

    order_type1 = get_order_type(1)
    direction1 = get_direction(1)

    order_type2 = get_order_type(2)
    direction2 = get_direction(2)
    

    print("\nYou have selected the following parameters for Poisson Process Data Generation:")
    print(f"Order Type 1: {order_type1}, Direction: {direction1}")
    print(f"Order Type 2: {order_type2}, Direction: {direction2}")


    order_counts, empirical_correlation, lambda_1, lambda_2 = aggregate_order_counts(df_message, order_type1, order_type2, direction1, direction2)
    df_simulated, simulated_corr = simulate_orders(empirical_correlation, lambda_1, lambda_2)

    print(f"Empirical Correlation: {empirical_correlation}")
    print(f"Simulated Correlation: {simulated_corr}")

    visualize_data(df_simulated, order_counts, order_type1, order_type2, empirical_correlation, simulated_corr)

    print(df_simulated)