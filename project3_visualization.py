# data_visualization_dashboard.py
"""
Data Visualization Dashboard
Processes scraped data and creates comprehensive visualizations
Demonstrates: pandas, matplotlib, seaborn, data analysis
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import numpy as np
import warnings
warnings.filterwarnings('ignore')

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)
plt.rcParams['font.size'] = 10

class DataVisualizationDashboard:
    def __init__(self, data_source):
        """
        Initialize dashboard with data
        
        Args:
            data_source (str or pd.DataFrame): CSV file path or DataFrame
        """
        if isinstance(data_source, str):
            self.df = pd.read_csv(data_source)
            print(f"Loaded {len(self.df)} records from {data_source}")
        else:
            self.df = data_source
            print(f"Loaded {len(self.df)} records from DataFrame")
        
        self._preprocess_data()
    
    def _preprocess_data(self):
        """Clean and prepare data for visualization"""
        # Remove duplicates
        original_len = len(self.df)
        self.df.drop_duplicates(inplace=True)
        if len(self.df) < original_len:
            print(f"Removed {original_len - len(self.df)} duplicate records")
        
        # Convert date columns if present
        date_columns = [col for col in self.df.columns if 'date' in col.lower() or 'time' in col.lower()]
        for col in date_columns:
            try:
                self.df[col] = pd.to_datetime(self.df[col])
                print(f"Converted {col} to datetime")
            except:
                pass
        
        print("\nData Info:")
        print(self.df.info())
        print("\nMissing Values:")
        print(self.df.isnull().sum())
    
    def create_summary_statistics(self):
        """Generate and display summary statistics"""
        print("\n" + "="*50)
        print("SUMMARY STATISTICS")
        print("="*50)
        
        # Basic stats
        print("\nDataset Shape:", self.df.shape)
        print("\nDescriptive Statistics:")
        print(self.df.describe())
        
        # Missing data analysis
        missing = self.df.isnull().sum()
        if missing.sum() > 0:
            print("\nMissing Data:")
            print(missing[missing > 0])
            print(f"\nTotal missing values: {missing.sum()} ({missing.sum()/self.df.size*100:.2f}%)")
    
    def plot_time_series(self, date_column, value_column, title="Time Series Analysis"):
        """
        Create time series visualization
        
        Args:
            date_column (str): Column containing dates
            value_column (str): Column with values to plot
            title (str): Plot title
        """
        fig, ax = plt.subplots(figsize=(14, 6))
        
        # Sort by date
        df_sorted = self.df.sort_values(date_column)
        
        # Plot line chart
        ax.plot(df_sorted[date_column], df_sorted[value_column], 
                marker='o', linewidth=2, markersize=4, alpha=0.7)
        
        # Add trend line
        z = np.polyfit(range(len(df_sorted)), df_sorted[value_column], 1)
        p = np.poly1d(z)
        ax.plot(df_sorted[date_column], p(range(len(df_sorted))), 
                "r--", alpha=0.5, linewidth=2, label='Trend')
        
        ax.set_xlabel(date_column.title(), fontsize=12, fontweight='bold')
        ax.set_ylabel(value_column.title(), fontsize=12, fontweight='bold')
        ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        filename = f'timeseries_{value_column}.png'
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"Saved: {filename}")
        plt.close()
    
    def plot_distribution(self, column, bins=30, title=None):
        """
        Create distribution histogram with KDE
        
        Args:
            column (str): Column to analyze
            bins (int): Number of bins
            title (str): Plot title
        """
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Remove NaN values
        data = self.df[column].dropna()
        
        # Plot histogram with KDE
        ax.hist(data, bins=bins, alpha=0.6, color='skyblue', edgecolor='black', density=True)
        
        # Add KDE
        data.plot(kind='kde', ax=ax, linewidth=2, color='darkblue')
        
        # Add statistics
        mean_val = data.mean()
        median_val = data.median()
        ax.axvline(mean_val, color='red', linestyle='--', linewidth=2, label=f'Mean: {mean_val:.2f}')
        ax.axvline(median_val, color='green', linestyle='--', linewidth=2, label=f'Median: {median_val:.2f}')
        
        ax.set_xlabel(column.title(), fontsize=12, fontweight='bold')
        ax.set_ylabel('Density', fontsize=12, fontweight='bold')
        ax.set_title(title or f'Distribution of {column.title()}', fontsize=14, fontweight='bold', pad=20)
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        
        filename = f'distribution_{column}.png'
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"Saved: {filename}")
        plt.close()
    
    def plot_category_comparison(self, category_column, value_column, top_n=10):
        """
        Create bar chart comparing categories
        
        Args:
            category_column (str): Categorical column
            value_column (str): Numeric column to aggregate
            top_n (int): Number of top categories to show
        """
        fig, ax = plt.subplots(figsize=(14, 8))
        
        # Aggregate data
        grouped = self.df.groupby(category_column)[value_column].agg(['mean', 'count']).reset_index()
        grouped = grouped.nlargest(top_n, 'mean')
        
        # Create bar plot
        bars = ax.barh(grouped[category_column], grouped['mean'], color='steelblue', alpha=0.8)
        
        # Add value labels
        for i, (bar, count) in enumerate(zip(bars, grouped['count'])):
            width = bar.get_width()
            ax.text(width, bar.get_y() + bar.get_height()/2, 
                   f' {width:.2f} (n={count})', 
                   va='center', fontweight='bold')
        
        ax.set_xlabel(f'Average {value_column.title()}', fontsize=12, fontweight='bold')
        ax.set_ylabel(category_column.title(), fontsize=12, fontweight='bold')
        ax.set_title(f'Top {top_n} {category_column.title()} by {value_column.title()}', 
                    fontsize=14, fontweight='bold', pad=20)
        ax.grid(True, alpha=0.3, axis='x')
        plt.tight_layout()
        
        filename = f'category_comparison_{category_column}.png'
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"Saved: {filename}")
        plt.close()
    
    def plot_correlation_heatmap(self, columns=None):
        """
        Create correlation heatmap for numeric columns
        
        Args:
            columns (list): Specific columns to include (optional)
        """
        # Select numeric columns
        if columns:
            numeric_df = self.df[columns]
        else:
            numeric_df = self.df.select_dtypes(include=[np.number])
        
        if numeric_df.empty:
            print("No numeric columns found for correlation")
            return
        
        # Calculate correlation matrix
        corr_matrix = numeric_df.corr()
        
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Create heatmap
        sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm', 
                   center=0, square=True, linewidths=1, cbar_kws={"shrink": 0.8}, ax=ax)
        
        ax.set_title('Correlation Heatmap', fontsize=14, fontweight='bold', pad=20)
        plt.tight_layout()
        
        filename = 'correlation_heatmap.png'
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"Saved: {filename}")
        plt.close()
    
    def plot_scatter_with_regression(self, x_column, y_column, hue_column=None):
        """
        Create scatter plot with regression line
        
        Args:
            x_column (str): X-axis column
            y_column (str): Y-axis column
            hue_column (str): Column for color coding (optional)
        """
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Create scatter plot
        if hue_column:
            categories = self.df[hue_column].unique()
            colors = sns.color_palette("husl", len(categories))
            for category, color in zip(categories, colors):
                mask = self.df[hue_column] == category
                ax.scatter(self.df.loc[mask, x_column], 
                          self.df.loc[mask, y_column],
                          label=category, alpha=0.6, s=50, color=color)
        else:
            ax.scatter(self.df[x_column], self.df[y_column], alpha=0.6, s=50, color='steelblue')
        
        # Add regression line
        mask = self.df[x_column].notna() & self.df[y_column].notna()
        z = np.polyfit(self.df.loc[mask, x_column], self.df.loc[mask, y_column], 1)
        p = np.poly1d(z)
        x_line = np.linspace(self.df[x_column].min(), self.df[x_column].max(), 100)
        ax.plot(x_line, p(x_line), "r--", linewidth=2, label='Regression Line')
        
        # Calculate R-squared
        correlation = self.df[x_column].corr(self.df[y_column])
        ax.text(0.05, 0.95, f'R² = {correlation**2:.3f}', 
               transform=ax.transAxes, fontsize=12, verticalalignment='top',
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        ax.set_xlabel(x_column.title(), fontsize=12, fontweight='bold')
        ax.set_ylabel(y_column.title(), fontsize=12, fontweight='bold')
        ax.set_title(f'{y_column.title()} vs {x_column.title()}', fontsize=14, fontweight='bold', pad=20)
        if hue_column:
            ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        
        filename = f'scatter_{x_column}_vs_{y_column}.png'
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"Saved: {filename}")
        plt.close()
    
    def create_dashboard(self, output_file='dashboard_summary.png'):
        """
        Create comprehensive dashboard with multiple subplots
        """
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns
        
        if len(numeric_cols) < 2:
            print("Not enough numeric columns for comprehensive dashboard")
            return
        
        fig = plt.figure(figsize=(16, 12))
        
        # Distribution plots
        for i, col in enumerate(numeric_cols[:4], 1):
            ax = plt.subplot(3, 2, i)
            self.df[col].hist(bins=30, ax=ax, alpha=0.7, color='skyblue', edgecolor='black')
            ax.set_title(f'Distribution: {col}', fontweight='bold')
            ax.set_xlabel(col)
            ax.set_ylabel('Frequency')
        
        # Box plot
        if len(numeric_cols) >= 2:
            ax = plt.subplot(3, 2, 5)
            self.df[numeric_cols[:4]].boxplot(ax=ax)
            ax.set_title('Box Plot Comparison', fontweight='bold')
            plt.xticks(rotation=45)
        
        # Summary stats
        ax = plt.subplot(3, 2, 6)
        ax.axis('off')
        summary_text = f"Dataset Summary\n{'='*30}\n"
        summary_text += f"Total Records: {len(self.df)}\n"
        summary_text += f"Features: {len(self.df.columns)}\n"
        summary_text += f"Missing Values: {self.df.isnull().sum().sum()}\n"
        summary_text += f"Duplicates: {self.df.duplicated().sum()}\n"
        ax.text(0.1, 0.5, summary_text, fontsize=12, family='monospace',
               verticalalignment='center')
        
        plt.tight_layout()
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"Saved dashboard: {output_file}")
        plt.close()


# Example usage
if __name__ == "__main__":
    # Create sample data for demonstration
    np.random.seed(42)
    sample_data = pd.DataFrame({
        'date': pd.date_range('2024-01-01', periods=100),
        'price': np.random.uniform(10, 100, 100),
        'quantity': np.random.randint(1, 50, 100),
        'category': np.random.choice(['A', 'B', 'C', 'D'], 100),
        'rating': np.random.uniform(1, 5, 100)
    })
    sample_data.to_csv('sample_data.csv', index=False)
    
    # Initialize dashboard
    dashboard = DataVisualizationDashboard('sample_data.csv')
    
    # Generate visualizations
    print("\nGenerating visualizations...")
    dashboard.create_summary_statistics()
    dashboard.plot_time_series('date', 'price', 'Price Over Time')
    dashboard.plot_distribution('price', bins=20)
    dashboard.plot_category_comparison('category', 'rating', top_n=4)
    dashboard.plot_correlation_heatmap()
    dashboard.plot_scatter_with_regression('quantity', 'price', 'category')
    dashboard.create_dashboard()
    
    print("\n✓ All visualizations created successfully!")