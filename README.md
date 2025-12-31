# Data Visualization Dashboard

A comprehensive Python toolkit for analyzing and visualizing scraped data with professional-grade charts, statistical analysis, and automated reporting.

## Features

- 📊 **Multiple Chart Types**: Time series, distributions, scatter plots, heatmaps, bar charts
- 📈 **Statistical Analysis**: Summary statistics, correlation analysis, trend detection
- 🎨 **Professional Styling**: Publication-ready visualizations with modern design
- 🔍 **Data Quality Checks**: Automatic detection of missing values, duplicates, outliers
- 🚀 **Automated Dashboards**: Generate comprehensive reports with one command
- 💾 **Flexible Input**: Supports CSV, JSON, and pandas DataFrames

## Technologies Used

- **Python 3.8+**
- **pandas** - Data manipulation and analysis
- **matplotlib** - Core plotting library
- **seaborn** - Statistical visualizations
- **numpy** - Numerical computations

## Installation

```bash
# Clone repository
git clone https://github.com/yourusername/data-visualization-dashboard.git
cd data-visualization-dashboard

# Install dependencies
pip install -r requirements.txt
```

## Requirements

Create `requirements.txt`:

```
pandas==2.1.4
matplotlib==3.8.2
seaborn==0.13.0
numpy==1.26.2
```

## Quick Start

### Basic Example

```python
from data_visualization_dashboard import DataVisualizationDashboard

# Load data from CSV
dashboard = DataVisualizationDashboard('scraped_data.csv')

# Generate summary statistics
dashboard.create_summary_statistics()

# Create visualizations
dashboard.plot_time_series('date', 'price', 'Price Trends Over Time')
dashboard.plot_distribution('rating', bins=20)
dashboard.plot_correlation_heatmap()

# Generate complete dashboard
dashboard.create_dashboard('full_report.png')
```

### From DataFrame

```python
import pandas as pd

# Load your scraped data
df = pd.read_csv('data.csv')

# Initialize dashboard
dashboard = DataVisualizationDashboard(df)

# Generate all visualizations
dashboard.create_dashboard()
```

## Available Visualizations

### 1. Time Series Analysis

Track trends over time with regression lines.

```python
dashboard.plot_time_series(
    date_column='date',
    value_column='price',
    title='Price Trends Over Time'
)
```

**Output**: `timeseries_price.png`

**Features**:
- Line plot with data points
- Trend line (linear regression)
- Automatic date formatting
- Grid and styling

### 2. Distribution Histogram

Analyze value distributions with density curves.

```python
dashboard.plot_distribution(
    column='price',
    bins=30,
    title='Price Distribution Analysis'
)
```

**Output**: `distribution_price.png`

**Features**:
- Histogram with customizable bins
- Kernel Density Estimation (KDE) overlay
- Mean and median lines
- Statistical annotations

### 3. Category Comparison

Compare metrics across categories with horizontal bar charts.

```python
dashboard.plot_category_comparison(
    category_column='product_type',
    value_column='rating',
    top_n=10
)
```

**Output**: `category_comparison_product_type.png`

**Features**:
- Shows top N categories
- Average values with count annotations
- Sorted by value
- Clean horizontal layout

### 4. Correlation Heatmap

Visualize relationships between numeric variables.

```python
dashboard.plot_correlation_heatmap(
    columns=['price', 'rating', 'sales', 'reviews']
)
```

**Output**: `correlation_heatmap.png`

**Features**:
- Color-coded correlation matrix
- Annotated correlation coefficients
- Symmetric layout
- Diverging colormap

### 5. Scatter Plot with Regression

Explore relationships between two variables.

```python
dashboard.plot_scatter_with_regression(
    x_column='price',
    y_column='sales',
    hue_column='category'  # Optional color coding
)
```

**Output**: `scatter_price_vs_sales.png`

**Features**:
- Scatter points (optionally color-coded)
- Linear regression line
- R² value annotation
- Category legends

### 6. Comprehensive Dashboard

Generate multi-panel report with key insights.

```python
dashboard.create_dashboard(output_file='report.png')
```

**Output**: `dashboard_summary.png`

**Features**:
- 6-panel layout
- Distribution histograms
- Box plot comparison
- Summary statistics panel

## Project Structure

```
data-visualization-dashboard/
├── data_visualization_dashboard.py  # Main dashboard class
├── requirements.txt                 # Python dependencies
├── README.md                        # Documentation
├── examples/
│   ├── ecommerce_analysis.py       # E-commerce data example
│   ├── time_series_demo.py         # Time series visualization
│   └── sales_report.py             # Sales dashboard example
├── sample_data/
│   ├── products.csv                # Sample dataset
│   └── sales_data.csv              # Sample time series
└── outputs/                         # Generated visualizations
    ├── timeseries_*.png
    ├── distribution_*.png
    └── dashboard_summary.png
```

## Complete Workflow Example

```python
from data_visualization_dashboard import DataVisualizationDashboard
import pandas as pd

# Step 1: Load scraped data
dashboard = DataVisualizationDashboard('scraped_ecommerce_data.csv')

# Step 2: Explore data quality
print("\n=== Data Quality Report ===")
dashboard.create_summary_statistics()

# Step 3: Time-based analysis
dashboard.plot_time_series(
    'scraped_date',
    'price',
    'Price Evolution Over Scraping Period'
)

# Step 4: Distribution analysis
dashboard.plot_distribution('rating', bins=20, title='Customer Ratings')
dashboard.plot_distribution('price', bins=30, title='Price Distribution')

# Step 5: Category comparison
dashboard.plot_category_comparison('brand', 'avg_rating', top_n=15)

# Step 6: Correlation analysis
dashboard.plot_correlation_heatmap([
    'price', 'rating', 'num_reviews', 'discount_pct'
])

# Step 7: Relationship exploration
dashboard.plot_scatter_with_regression(
    'num_reviews',
    'rating',
    hue_column='price_category'
)

# Step 8: Generate comprehensive report
dashboard.create_dashboard('final_report.png')

print("\n✓ Analysis complete! Check outputs/ folder for visualizations.")
```

## Customization

### Styling

Customize plot appearance:

```python
import matplotlib.pyplot as plt
import seaborn as sns

# Change color palette
sns.set_palette("husl")

# Modify default figure size
plt.rcParams['figure.figsize'] = (16, 10)

# Change font
plt.rcParams['font.family'] = 'Arial'
plt.rcParams['font.size'] = 12

# Then create dashboard
dashboard = DataVisualizationDashboard('data.csv')
```

### Custom Preprocessing

Override preprocessing method:

```python
class CustomDashboard(DataVisualizationDashboard):
    def _preprocess_data(self):
        super()._preprocess_data()
        
        # Custom cleaning
        self.df['price'] = self.df['price'].str.replace('$', '').astype(float)
        self.df['rating'] = pd.to_numeric(self.df['rating'], errors='coerce')
        
        # Feature engineering
        self.df['price_category'] = pd.cut(
            self.df['price'],
            bins=[0, 20, 50, 100, float('inf')],
            labels=['Budget', 'Mid', 'Premium', 'Luxury']
        )
```

## Advanced Features

### 1. Outlier Detection

```python
# Use box plots to identify outliers
dashboard.create_dashboard()  # Includes box plot panel

# Or manually with IQR method
Q1 = dashboard.df['price'].quantile(0.25)
Q3 = dashboard.df['price'].quantile(0.75)
IQR = Q3 - Q1
outliers = dashboard.df[
    (dashboard.df['price'] < Q1 - 1.5 * IQR) |
    (dashboard.df['price'] > Q3 + 1.5 * IQR)
]
print(f"Outliers detected: {len(outliers)}")
```

### 2. Missing Data Visualization

```python
import matplotlib.pyplot as plt

# Visualize missing data
missing = dashboard.df.isnull().sum()
missing = missing[missing > 0].sort_values(ascending=False)

plt.figure(figsize=(10, 6))
missing.plot(kind='bar', color='coral')
plt.title('Missing Values by Column', fontweight='bold', fontsize=14)
plt.ylabel('Count')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('missing_data_analysis.png', dpi=300)
```

### 3. Multi-Variable Comparison

```python
# Pairplot for multiple variables
import seaborn as sns

numeric_cols = dashboard.df.select_dtypes(include=[np.number]).columns[:4]
sns.pairplot(dashboard.df[numeric_cols], diag_kind='kde', corner=True)
plt.savefig('pairplot_analysis.png', dpi=300, bbox_inches='tight')
```

### 4. Time Series Decomposition

```python
from statsmodels.tsa.seasonal import seasonal_decompose

# Decompose time series
decomposition = seasonal_decompose(
    dashboard.df.set_index('date')['price'],
    model='additive',
    period=7
)

fig, axes = plt.subplots(4, 1, figsize=(12, 10))
decomposition.observed.plot(ax=axes[0], title='Observed')
decomposition.trend.plot(ax=axes[1], title='Trend')
decomposition.seasonal.plot(ax=axes[2], title='Seasonal')
decomposition.resid.plot(ax=axes[3], title='Residual')
plt.tight_layout()
plt.savefig('time_series_decomposition.png', dpi=300)
```

## Sample Outputs

### Time Series Plot
![Time Series Example](https://via.placeholder.com/800x400?text=Price+Trends+Over+Time)

### Distribution Histogram
![Distribution Example](https://via.placeholder.com/800x400?text=Price+Distribution+with+KDE)

### Correlation Heatmap
![Heatmap Example](https://via.placeholder.com/800x400?text=Correlation+Matrix)

## Performance Tips

1. **Handle large datasets efficiently**:
   ```python
   # Sample large datasets
   if len(df) > 100000:
       df = df.sample(n=10000, random_state=42)
   ```

2. **Optimize memory usage**:
   ```python
   # Use categorical dtype for string columns
   df['category'] = df['category'].astype('category')
   ```

3. **Batch export**:
   ```python
   # Save all plots without displaying
   plt.ioff()  # Turn off interactive mode
   dashboard.create_dashboard()
   plt.close('all')
   ```

## Use Cases

### E-commerce Analysis
- Price trends over time
- Rating distributions by product category
- Correlation between price and reviews

### Content Scraping
- Publication frequency analysis
- Author contribution comparison
- Topic trend visualization

### Real Estate Data
- Price per square foot distributions
- Location-based comparisons
- Time-to-sell analysis

### Social Media Analytics
- Engagement metrics over time
- Post performance by category
- Follower growth trends

## Troubleshooting

### Issue: Plots not displaying
```python
# Add at end of script
plt.show()

# Or save explicitly
plt.savefig('output.png')
```

### Issue: Memory errors with large datasets
```python
# Downsample data
df_sample = df.sample(frac=0.1, random_state=42)
dashboard = DataVisualizationDashboard(df_sample)
```

### Issue: Font warnings
```python
# Suppress warnings
import warnings
warnings.filterwarnings('ignore')
```

## Best Practices

✅ Always check data quality first (`create_summary_statistics()`)  
✅ Use appropriate chart types for data characteristics  
✅ Include clear titles and axis labels  
✅ Save high-resolution images (dpi=300) for reports  
✅ Document data sources and preprocessing steps  
✅ Use consistent color schemes across visualizations  

## Statistical Outputs

The dashboard provides comprehensive statistics:

```
Dataset Summary
==================
Total Records: 1,250
Features: 8
Missing Values: 15 (0.15%)
Duplicates: 3

Descriptive Statistics:
       price  rating  reviews
count  1247   1245    1250
mean   45.23  4.12    156.8
std    23.45  0.87    234.2
min    9.99   1.0     0
25%    24.99  3.5     45
50%    39.99  4.2     98
75%    59.99  4.8     189
max    199.99 5.0     2341
```

## Future Enhancements

- [ ] Interactive dashboards with Plotly
- [ ] Automated anomaly detection
- [ ] Machine learning insights integration
- [ ] PDF report generation
- [ ] Real-time streaming visualization
- [ ] Geographic mapping capabilities

## Contributing

Contributions welcome! Please:
1. Fork the repository
2. Create feature branch
3. Add tests for new visualizations
4. Submit pull request

## License

MIT License

## Contact

- GitHub: [@GreatestCodeMaster](https://github.com/waibazen)
- Email: lamaprahlad5@gmail.com

## Acknowledgments

Built for analyzing web-scraped data with professional visualization standards suitable for data science portfolios.

---

**Note**: This toolkit is designed to complement web scraping projects. Always ensure proper data rights and privacy compliance when visualizing scraped data.
