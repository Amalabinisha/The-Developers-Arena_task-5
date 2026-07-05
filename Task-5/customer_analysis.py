import os
import itertools
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from fpdf import FPDF

sns.set_theme(style="whitegrid")
plt.rcParams['figure.figsize'] = (10, 6)
os.makedirs('plots', exist_ok=True)

print("🔍 --- STEP 1: LOADING & EXPLORING ACTUAL DATASETS ---")

try:
    sales_df = pd.read_csv('sales_data.csv')
    customer_df = pd.read_csv('customer_churn.csv')
    print("✅ Source datasets loaded successfully.")
except FileNotFoundError as e:
    print(f"❌ Error: Ensure your CSV files are in the working directory! Details: {e}")
    raise

for name, df in [("SALES SOURCE", sales_df), ("CUSTOMER CHURN SOURCE", customer_df)]:
    print(f"\n===== PROFILING METADATA FOR: {name} =====")
    print(f"Shape: {df.shape[0]} rows, {df.shape[1]} columns")
    print("\n--- Missing Value Counts ---")
    print(df.isnull().sum())
    print("-" * 40)

print("\n🧹 --- STEP 2: CLEANING & UNIFYING DATA SCHEMAS ---")

sales_df.rename(columns={'Customer_ID': 'CustomerID', 'Total_Sales': 'TotalAmount'}, inplace=True)

sales_df['Date'] = pd.to_datetime(sales_df['Date'], format='%d-%m-%Y', exact=False, errors='coerce')
sales_df['Date'] = sales_df['Date'].fillna(datetime(2026, 1, 1))

sales_df['Product'] = sales_df['Product'].astype(str).str.strip().str.upper()
sales_df['Region'] = sales_df['Region'].astype(str).str.strip().str.upper()

sales_df['Year'] = sales_df['Date'].dt.year
sales_df['Month'] = sales_df['Date'].dt.strftime('%B')
sales_df['MonthNum'] = sales_df['Date'].dt.month
sales_df['Day'] = sales_df['Date'].dt.day

customer_df['TotalCharges'] = pd.to_numeric(customer_df['TotalCharges'], errors='coerce').fillna(0)

print("✅ Data structures transformed and datetime metrics mapped.")

print("\n🔄 --- STEP 3: EXECUTING RELATIONAL MERGE & CORE CALCULATIONS ---")

merged_df = pd.merge(sales_df, customer_df, on='CustomerID', how='left')

merged_df['Region'] = merged_df['Region'].fillna('UNKNOWN REGION')
merged_df['Churn'] = merged_df['Churn'].fillna(0).astype(int)

customer_metrics = merged_df.groupby(['CustomerID', 'Region']).agg(
    Customer_Lifetime_Value=('TotalAmount', 'sum'),
    Total_Orders_Placed=('Quantity', 'count'),
    Average_Order_Size=('Quantity', 'mean')
).reset_index()

total_revenue = sales_df['TotalAmount'].sum()
total_unique_customers = customer_df['CustomerID'].nunique()
average_order_value = sales_df['TotalAmount'].mean()

if not customer_metrics.empty:
    top_client_record = customer_metrics.sort_values(by='Customer_Lifetime_Value', ascending=False).iloc[0]
    top_customer_ident = top_client_record['CustomerID']
    top_customer_clv = top_client_record['Customer_Lifetime_Value']
else:
    top_customer_ident, top_customer_clv = "N/A", 0

total_accounts = len(customer_df)
churned_accounts = customer_df['Churn'].eq(1).sum()
active_accounts = total_accounts - churned_accounts
calculated_retention_rate = (active_accounts / total_accounts) * 100 if total_accounts > 0 else 0

print(f"📊 Summary: Total Rev: ${total_revenue:,.2f} | Retention Rate: {calculated_retention_rate:.2f}%")

print("\n🧮 --- STEP 4: RUNNING CROSS-SELLING AND PIVOT MATRIX CALCULATIONS ---")

customer_baskets = merged_df.groupby('CustomerID')['Product'].apply(list)
co_occurrence_dict = {}

for basket in customer_baskets:
    if len(basket) > 1:
        unique_combinations = sorted(list(set(itertools.combinations(sorted(basket), 2))))
        for combo in unique_combinations:
            co_occurrence_dict[combo] = co_occurrence_dict.get(combo, 0) + 1

if co_occurrence_dict:
    cross_sell_df = pd.DataFrame([
        {'Product_A': k[0], 'Product_B': k[1], 'Co_Purchase_Frequency': v} 
        for k, v in co_occurrence_dict.items()
    ]).sort_values(by='Co_Purchase_Frequency', ascending=False).reset_index(drop=True)
else:
    cross_sell_df = pd.DataFrame(columns=['Product_A', 'Product_B', 'Co_Purchase_Frequency'])

high_val_active_mask = (merged_df['TotalAmount'] >= 50000) & (merged_df['Churn'] == 0)
high_val_active_df = merged_df[high_val_active_mask]

strategic_focus_mask = (merged_df['Region'] == 'EAST') | (merged_df['Quantity'] >= 5)
strategic_focus_df = merged_df[strategic_focus_mask]

regional_product_matrix = pd.pivot_table(
    merged_df,
    values='TotalAmount',
    index='Region',
    columns='Product',
    aggfunc='sum',
    fill_value=0
)

print("✅ Advanced analytics processing complete.")

print("\n🎨 --- STEP 5: GENERATING DASHBOARD VISUALIZATIONS ---")

plt.figure(figsize=(10, 5))
top_accounts = customer_metrics.sort_values(by='Customer_Lifetime_Value', ascending=False).head(5)
sns.barplot(data=top_accounts, x='Customer_Lifetime_Value', y='CustomerID', palette='viridis')
plt.title('Top Valuable Customer IDs by Sales Generation Profile', fontsize=13, fontweight='bold', pad=12)
plt.xlabel('Customer Value Output ($)')
plt.ylabel('Customer ID Label')
plt.tight_layout()
plt.savefig('plots/customer_clv_ranking.png', dpi=300)
plt.close()

plt.figure(figsize=(10, 5))
sns.heatmap(regional_product_matrix, annot=True, fmt=".0f", cmap='Blues', cbar_kws={'label': 'Gross Revenue ($)'})
plt.title('Regional Performance Matrix Across Product Segments', fontsize=13, fontweight='bold', pad=12)
plt.xlabel('Product Categories')
plt.ylabel('Sales Territory Region')
plt.tight_layout()
plt.savefig('plots/regional_density_heatmap.png', dpi=300)
plt.close()

plt.figure(figsize=(10, 5))
if not cross_sell_df.empty:
    cross_sell_df['Pairing'] = cross_sell_df['Product_A'] + " + " + cross_sell_df['Product_B']
    sns.barplot(data=cross_sell_df.head(5), x='Co_Purchase_Frequency', y='Pairing', palette='rocket')
    plt.title('Product Bundle Association Frequencies (Cross-Selling Engine)', fontsize=13, fontweight='bold', pad=12)
    plt.xlabel('Observed Co-Purchase Frequency Counts')
    plt.ylabel('Product Package Grouping')
else:
    plt.text(0.5, 0.5, 'No Multi-Item Purchase Combinations Detected Across Customer Cohorts', ha='center', va='center', fontsize=11, color='gray')
    plt.title('Product Bundle Association Frequencies', fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig('plots/cross_selling_affinity.png', dpi=300)
plt.close()

plt.figure(figsize=(6, 6))
portfolio_mix = [active_accounts, churned_accounts]
plt.pie(portfolio_mix, labels=['Active Client Base', 'Churned Portfolio'], autopct='%1.1f%%', 
        colors=['#4CAF50', '#FF5722'], startangle=140, explode=(0, 0.05))
plt.title('Enterprise Account Portfolio Retention Health Mix', fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig('plots/portfolio_retention_pie.png', dpi=300)
plt.close()

print("✅ Performance graphs plotted and exported inside the 'plots/' subfolder.")

print("\n📝 --- STEP 6: EXPORTING STRATEGIC COMPREHENSIVE REVIEWS TO PDF ---")

class ExecutiveReportFormat(FPDF):
    def header(self):
        self.set_fill_color(24, 43, 73)  
        self.rect(0, 0, 210, 32, 'F')
        self.set_text_color(255, 255, 255)
        self.set_font('Helvetica', 'B', 15)
        # Modern fpdf2 structure formatting parameters used here
        self.cell(0, 5, 'CUSTOMER PURCHASING PATTERNS & SALES PERFORMANCE', 0, align='C', new_x="LMARGIN", new_y="NEXT")
        self.set_font('Helvetica', 'I', 9)
        self.cell(0, 7, f'Automated Executive Insights Dashboard | Compiled Date: {datetime.now().strftime("%Y-%m-%d")}', 0, align='C', new_x="LMARGIN", new_y="NEXT")
        self.ln(12)

    def footer(self):
        self.set_y(-15)
        self.set_font('Helvetica', 'I', 8)
        self.set_text_color(130, 130, 130)
        self.cell(0, 10, f'Internal Business Intelligence Summary - Page {self.page_no()}', 0, 0, 'C')

    def insert_section_banner(self, banner_title):
        self.set_font('Helvetica', 'B', 12)
        self.set_text_color(24, 43, 73)
        self.cell(0, 8, banner_title, 0, align='L', new_x="LMARGIN", new_y="NEXT")
        self.line(self.get_x(), self.get_y(), self.get_x() + 190, self.get_y())
        self.ln(3)

report = ExecutiveReportFormat()
report.set_auto_page_break(auto=True, margin=15)
report.add_page()

report.insert_section_banner("1. High-Level Performance Aggregations")
report.set_font('Helvetica', '', 10)
report.set_text_color(40, 40, 40)

# FIXED: Replaced unsafe unicode dot bullet points with clean ASCII dashes to avoid encoding crashes
report.cell(0, 6, f"- Total Realized Revenue Portfolio Base: ${total_revenue:,.2f}", 0, new_x="LMARGIN", new_y="NEXT")
report.cell(0, 6, f"- System Monitored Enterprise Customer Base: {total_unique_customers} profiles", 0, new_x="LMARGIN", new_y="NEXT")
report.cell(0, 6, f"- Mean Transaction Order Value (AOV Performance Index): ${average_order_value:,.2f}", 0, new_x="LMARGIN", new_y="NEXT")
report.cell(0, 6, f"- Evaluated Base Customer Account Retention Score: {calculated_retention_rate:.2f}%", 0, new_x="LMARGIN", new_y="NEXT")
report.cell(0, 6, f"- Identified High-Velocity Account Target: Customer ID {top_customer_ident} (${top_customer_clv:,.2f})", 0, new_x="LMARGIN", new_y="NEXT")
report.ln(5)

report.insert_section_banner("2. Operational Revenue Charts & Retention Summarizations")
report.image('plots/customer_clv_ranking.png', x=15, w=180, h=75)
report.ln(3)

report.add_page()
report.insert_section_banner("3. Regional Demand Matrices Analysis")
report.image('plots/regional_density_heatmap.png', x=15, w=180, h=80)
report.ln(4)

report.insert_section_banner("4. Account Loyalty Distributions Breakdown")
report.image('plots/portfolio_retention_pie.png', x=55, w=100, h=100)
report.ln(5)

report.add_page()
report.insert_section_banner("5. Strategic Insights & Executive Recommendations")
report.set_font('Helvetica', '', 10)

business_recommendations = [
    f"Cross-Product Optimization: Leverage market basket pairing configurations to deploy targeted bundles during checkout processes, optimizing under-utilized product pairings.",
    f"High-Value Account Security: Assign localized account management workflows to stabilize high-yield customer profiles. Customer {top_customer_ident} represents a high-priority asset generating ${top_customer_clv:,.2f}.",
    f"Churn Prevention Infrastructure: A calculated customer retention rate of {calculated_retention_rate:.2f}% calls for automated proactive campaigns tailored specifically toward Month-to-Month contracts with high monthly charge rates.",
    f"Territorial Growth Allocations: Reallocate digital marketing expenditure from standard pipelines directly into your high-performing product regions identified in your regional heatmap matrices."
]

for instruction in business_recommendations:
    report.multi_cell(0, 6.5, f"- {instruction}")
    report.ln(2)

report.output('analysis_report.pdf')

print("\n🏆 SUCCESS: All execution steps run perfectly! Output deliverables 'analysis_report.pdf' and graphs inside 'plots/' are finalized.")