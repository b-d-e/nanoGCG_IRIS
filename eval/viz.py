import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from collections import OrderedDict
import seaborn as sns

# Set paper-quality styling
plt.rcParams.update({
    'font.size': 12,
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'DejaVu Serif'],
    'axes.linewidth': 1.2,
    'axes.spines.left': True,
    'axes.spines.bottom': True,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'xtick.direction': 'in',
    'ytick.direction': 'in',
    'axes.grid': True,
    'grid.alpha': 0.3,
    'grid.linewidth': 0.8,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.facecolor': 'white'
})

def load_ranking_data(csv_path, experiment_name):
    """Load and validate ranking data for a single experiment"""
    try:
        df = pd.read_csv(csv_path)
        print(f"✅ Loaded {len(df)} rows from {experiment_name}: {csv_path}")
        
        if 'ranking' not in df.columns:
            raise ValueError(f"No 'ranking' column found in {experiment_name}")
        
        # Filter out unranked items
        ranked_df = df[df['ranking'].notna()].copy()
        print(f"📊 Found {len(ranked_df)} ranked responses for {experiment_name}")
        
        if len(ranked_df) == 0:
            raise ValueError(f"No ranked responses found in {experiment_name}")
        
        return ranked_df
        
    except Exception as e:
        print(f"❌ Error loading {experiment_name}: {e}")
        return None

def analyze_rankings(df):
    """Analyze ranking distribution for a single experiment"""
    ranking_counts = df['ranking'].value_counts()
    total_ranked = len(df)
    
    # Define standard categories - using paper-appropriate colors
    categories = {
        'Refusal': {'color': '#d62728', 'count': 0},           # Red
        'Off-task': {'color': '#7f7f7f', 'count': 0},          # Grey 
        'Successful Attack': {'color': '#2ca02c', 'count': 0}   # Green
    }
    
    # Update counts from actual data
    for category in ranking_counts.index:
        if category in categories:
            categories[category]['count'] = ranking_counts[category]
        else:
            print(f"⚠️  Unknown category found: {category}")
    
    # Calculate percentages
    for category in categories:
        count = categories[category]['count']
        categories[category]['percentage'] = (count / total_ranked) * 100 if total_ranked > 0 else 0
    
    return categories, total_ranked

def load_multiple_experiments(experiment_dict):
    """Load data for multiple experiments"""
    experiments_data = OrderedDict()
    
    for exp_name, csv_path in experiment_dict.items():
        if not Path(csv_path).exists():
            print(f"❌ File not found for {exp_name}: {csv_path}")
            continue
            
        df = load_ranking_data(csv_path, exp_name)
        if df is not None:
            categories, total_ranked = analyze_rankings(df)
            experiments_data[exp_name] = {
                'categories': categories,
                'total_ranked': total_ranked,
                'dataframe': df
            }
    
    return experiments_data

def create_comparison_chart(experiments_data, output_path=None, 
                          title="Model Response Classification Comparison",
                          figsize=(14, 8)):
    """Create paper-quality comparison chart with stacked horizontal bars"""
    
    if not experiments_data:
        print("❌ No experiment data to plot")
        return None
    
    # Prepare data
    experiment_names = list(experiments_data.keys())
    n_experiments = len(experiment_names)
    
    # Category information
    category_info = {
        'Refusal': {'color': '#d62728', 'label': 'Refusal'},
        'Off-task': {'color': '#7f7f7f', 'label': 'Off-task'},
        'Successful Attack': {'color': '#2ca02c', 'label': 'Successful Attack'}
    }
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Set up positions
    y_positions = np.arange(n_experiments)
    bar_height = 0.6
    
    # Plot each experiment
    for i, (exp_name, exp_data) in enumerate(experiments_data.items()):
        categories = exp_data['categories']
        total_ranked = exp_data['total_ranked']
        
        # Calculate cumulative positions
        cumulative = 0
        
        for category_name, category_props in category_info.items():
            if category_name in categories and categories[category_name]['count'] > 0:
                count = categories[category_name]['count']
                percentage = (count / total_ranked) * 100
                
                # Create bar segment
                ax.barh(y_positions[i], percentage, left=cumulative, 
                       height=bar_height, color=category_props['color'], 
                       alpha=0.85, edgecolor='white', linewidth=1)
                
                # Add count labels (if space allows)
                if percentage > 8:  # Only show if segment is wide enough
                    ax.text(cumulative + percentage/2, y_positions[i], 
                           f'{count}', ha='center', va='center', 
                           fontweight='bold', fontsize=10, color='white')
                
                cumulative += percentage
    
    # Customize chart
    ax.set_xlim(0, 100)
    ax.set_ylim(-0.5, n_experiments - 0.5)
    
    # Set y-axis
    ax.set_yticks(y_positions)
    ax.set_yticklabels(experiment_names, fontsize=11)
    ax.invert_yaxis()  # Top to bottom
    
    # Set x-axis
    ax.set_xlabel('Percentage of Responses (%)', fontsize=12, fontweight='bold')
    ax.set_xticks(range(0, 101, 20))
    ax.set_xticklabels([f"{x}" for x in range(0, 101, 20)])
    
    # Title
    ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
    
    # Grid
    ax.grid(axis='x', alpha=0.3, linestyle='-', linewidth=0.8)
    ax.set_axisbelow(True)
    
    # Create legend
    legend_elements = []
    for category_name, category_props in category_info.items():
        legend_elements.append(plt.Rectangle((0,0), 1, 1, 
                                           facecolor=category_props['color'], 
                                           alpha=0.85,
                                           label=category_props['label']))
    
    ax.legend(handles=legend_elements, loc='lower right', 
              frameon=True, fancybox=True, shadow=True, 
              fontsize=11, title='Response Types', title_fontsize=12)
    
    plt.tight_layout()
    
    # Save if path provided
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight', 
                   facecolor='white', edgecolor='none')
        print(f"💾 Comparison chart saved to: {output_path}")
    
    return fig, ax

def create_summary_table(experiments_data):
    """Create a summary table of results"""
    print("\n" + "="*80)
    print("📊 MULTI-EXPERIMENT COMPARISON SUMMARY")
    print("="*80)
    
    # Headers
    print(f"{'Experiment':.<25} {'Total':>8} {'Refusal':>10} {'Off-task':>10} {'Success':>10} {'Success %':>10}")
    print("-" * 80)
    
    for exp_name, exp_data in experiments_data.items():
        categories = exp_data['categories']
        total = exp_data['total_ranked']
        
        refusal_count = categories['Refusal']['count']
        offtask_count = categories['Off-task']['count']
        success_count = categories['Successful Attack']['count']
        success_pct = categories['Successful Attack']['percentage']
        
        print(f"{exp_name:.<25} {total:>8} {refusal_count:>10} {offtask_count:>10} "
              f"{success_count:>10} {success_pct:>9.1f}%")
    
    print("="*80)

def create_success_rate_comparison(experiments_data, output_path=None):
    """Create a focused comparison of success rates"""
    experiment_names = list(experiments_data.keys())
    success_rates = []
    
    for exp_data in experiments_data.values():
        success_rates.append(exp_data['categories']['Successful Attack']['percentage'])
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Create bar chart
    bars = ax.bar(range(len(experiment_names)), success_rates, 
                  color='#2ca02c', alpha=0.7, edgecolor='black', linewidth=1.2)
    
    # Add value labels on bars
    for i, (bar, rate) in enumerate(zip(bars, success_rates)):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                f'{rate:.1f}%', ha='center', va='bottom', fontweight='bold')
    
    # Customize
    ax.set_xlabel('Experiment', fontsize=12, fontweight='bold')
    ax.set_ylabel('Attack Success Rate (%)', fontsize=12, fontweight='bold')
    ax.set_title('Attack Success Rate Comparison', fontsize=14, fontweight='bold', pad=20)
    ax.set_xticks(range(len(experiment_names)))
    ax.set_xticklabels(experiment_names, rotation=45, ha='right')
    ax.set_ylim(0, max(success_rates) * 1.1 + 5)
    
    # Grid
    ax.grid(axis='y', alpha=0.3, linestyle='-', linewidth=0.8)
    ax.set_axisbelow(True)
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight', 
                   facecolor='white', edgecolor='none')
        print(f"💾 Success rate comparison saved to: {output_path}")
    
    return fig, ax

def main_comparison(experiment_dict, output_dir="./", base_filename="experiment_comparison"):
    """Main function for multi-experiment comparison"""
    
    # Create output directory if it doesn't exist
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Load all experiments
    print("🔄 Loading experiment data...")
    experiments_data = load_multiple_experiments(experiment_dict)
    
    if not experiments_data:
        print("❌ No valid experiment data found")
        return
    
    print(f"\n✅ Successfully loaded {len(experiments_data)} experiments")
    
    # Create summary table
    create_summary_table(experiments_data)
    
    # Generate visualizations
    comparison_chart_path = output_path / f"{base_filename}_comparison.png"
    success_rate_chart_path = output_path / f"{base_filename}_success_rates.png"
    
    print(f"\n📈 Creating comparison chart...")
    fig1, ax1 = create_comparison_chart(experiments_data, comparison_chart_path)
    
    print(f"📊 Creating success rate comparison...")
    fig2, ax2 = create_success_rate_comparison(experiments_data, success_rate_chart_path)
    
    # Show plots
    plt.show()
    
    print(f"\n✅ Multi-experiment analysis complete!")
    print(f"📁 Files saved to: {output_path}")

# Example usage
if __name__ == "__main__":
    # Define your experiments - modify these paths as needed
    experiments = OrderedDict([
        # ("Vanilla GCG", "rankings/vanilla_gcg.csv"),
        # ("GCG + Refusal Direction", "rankings/gcg_refusal_dir.csv"),
        # ("Force Orthogonal", "rankings/force_ortho.csv"),
        # ("Baseline", "rankings/baseline.csv")
        ("No Attack", "rankings/base_model.csv"),
        ("Naive GCG - '</think>\\n\\nSure'", "rankings/naive_gcg.csv"),
        ("Force Ortho Only", "rankings/force_ortho.csv"),
        ("Beta 0.7", "rankings/beta0.7.csv"),
        ("Beta 0.9", "rankings/beta0.9.csv"),
    ])
    
    # Run comparison
    main_comparison(experiments, output_dir="./figures/", 
                   base_filename="gcg_experiments")