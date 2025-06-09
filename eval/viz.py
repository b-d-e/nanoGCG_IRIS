import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from collections import OrderedDict
import seaborn as sns
import re
from scipy.interpolate import griddata
from scipy.interpolate import interp2d

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

def extract_beta_m_parameters(experiment_name):
    """Extract beta and m parameters from experiment name
    
    Returns:
        tuple: (beta, m) or (None, None) if not extractable
    """
    # Handle the orthogonalisation only case (β=0, m=0)
    if "Force orthogonalised output only" in experiment_name and "β=0" in experiment_name:
        return (0.0, 0)
    
    # Extract beta parameter
    beta_match = re.search(r'β=([0-9.]+)', experiment_name)
    beta = float(beta_match.group(1)) if beta_match else None
    
    # Extract m parameter  
    m_match = re.search(r'm=([0-9]+)', experiment_name)
    m = int(m_match.group(1)) if m_match else None
    
    # Special case: if we have beta but no m, and it's orthogonalisation only
    if beta == 0.0 and m is None and "Force orthogonalised output only" in experiment_name:
        m = 0
    
    return (beta, m)

def analyze_rankings(df):
    """Analyze ranking distribution for a single experiment"""
    ranking_counts = df['ranking'].value_counts()
    total_ranked = len(df)
    
    # Define standard categories - using consistent colors from examples
    categories = {
        'Off-task': {'color': '#808080', 'count': 0},          
        'Successful Attack': {'color': '#6b8e23', 'count': 0} ,
        'Refusal': {'color': '#DAE8FC', 'count': 0}
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
            
            # Extract parameters
            beta, m = extract_beta_m_parameters(exp_name)
            
            experiments_data[exp_name] = {
                'categories': categories,
                'total_ranked': total_ranked,
                'dataframe': df,
                'beta': beta,
                'm': m
            }
    
    return experiments_data

def create_parameter_heatmap(experiments_data, output_path=None, 
                           figsize=(10, 8), interpolation_method='linear'):
    """Create heatmap of ASR vs beta and m parameters
    
    Args:
        interpolation_method: 'linear', 'cubic', or 'nearest' for scipy.interpolate.griddata
    """
    
    # Filter experiments with valid beta and m parameters
    param_experiments = {name: data for name, data in experiments_data.items() 
                        if data['beta'] is not None and data['m'] is not None}
    
    if len(param_experiments) < 3:
        print("❌ Need at least 3 experiments with beta/m parameters for meaningful heatmap")
        return None, None
    
    # Extract data points
    betas = []
    ms = []
    asrs = []
    
    print("\n📊 Parameter extraction for heatmap:")
    for exp_name, exp_data in param_experiments.items():
        beta = exp_data['beta']
        m = exp_data['m']
        asr = exp_data['categories']['Successful Attack']['percentage']
        
        betas.append(beta)
        ms.append(m)
        asrs.append(asr)
        
        print(f"  {exp_name}: β={beta}, m={m}, ASR={asr:.1f}%")
    
    betas = np.array(betas)
    ms = np.array(ms)
    asrs = np.array(asrs)
    
    # Create interpolation grid
    beta_min, beta_max = betas.min(), betas.max()
    m_min, m_max = ms.min(), ms.max()
    
    # Extend ranges slightly for better visualization
    beta_range = beta_max - beta_min
    m_range = m_max - m_min
    
    beta_grid = np.linspace(beta_min - 0.1 * beta_range, beta_max + 0.1 * beta_range, 50)
    m_grid = np.linspace(m_min - 0.1 * m_range, m_max + 0.1 * m_range, 50)
    
    beta_mesh, m_mesh = np.meshgrid(beta_grid, m_grid)
    
    # Interpolate ASR values
    try:
        asr_interpolated = griddata((betas, ms), asrs, (beta_mesh, m_mesh), method=interpolation_method)
    except Exception as e:
        print(f"⚠️  Interpolation failed with {interpolation_method}, falling back to 'nearest': {e}")
        asr_interpolated = griddata((betas, ms), asrs, (beta_mesh, m_mesh), method='nearest')
    
    # Create heatmap
    fig, ax = plt.subplots(figsize=figsize)
    
    # Create heatmap with custom colormap
    im = ax.imshow(asr_interpolated, extent=[beta_min - 0.1 * beta_range, beta_max + 0.1 * beta_range,
                                          m_min - 0.1 * m_range, m_max + 0.1 * m_range],
                  origin='lower', aspect='auto', cmap='RdYlBu_r', alpha=0.8)
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Attack Success Rate (%)', fontsize=12)
    
    # Overlay actual data points
    scatter = ax.scatter(betas, ms, c=asrs, s=150, cmap='RdYlBu_r', 
                        edgecolors='black', linewidth=2, zorder=5)
    
    # Add data point labels
    for i, (beta, m, asr) in enumerate(zip(betas, ms, asrs)):
        ax.annotate(f'{asr:.1f}%', (beta, m), xytext=(5, 5), 
                   textcoords='offset points', fontsize=10, fontweight='bold',
                   bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))
    
    # Customize plot
    ax.set_xlabel('β (Loss term balancing)', fontsize=14)
    ax.set_ylabel('m (Num CoT Tokens)', fontsize=14)
    ax.set_title('Attack Success Rate Heatmap\n(β vs m parameters)', fontsize=16)
    
    # Add grid
    ax.grid(True, alpha=0.3, linestyle='--')
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"💾 Parameter heatmap saved to: {output_path}")
    
    return fig, ax

def create_parameter_contour(experiments_data, output_path=None, 
                           figsize=(10, 8), n_levels=10, interpolation_method='linear'):
    """Create contour plot of ASR vs beta and m parameters"""
    
    # Filter experiments with valid beta and m parameters
    param_experiments = {name: data for name, data in experiments_data.items() 
                        if data['beta'] is not None and data['m'] is not None}
    
    if len(param_experiments) < 3:
        print("❌ Need at least 3 experiments with beta/m parameters for meaningful contour plot")
        return None, None
    
    # Extract data points
    betas = []
    ms = []
    asrs = []
    
    for exp_name, exp_data in param_experiments.items():
        betas.append(exp_data['beta'])
        ms.append(exp_data['m'])
        asrs.append(exp_data['categories']['Successful Attack']['percentage'])
    
    betas = np.array(betas)
    ms = np.array(ms)
    asrs = np.array(asrs)
    
    # Create interpolation grid
    beta_min, beta_max = betas.min(), betas.max()
    m_min, m_max = ms.min(), ms.max()
    
    # Extend ranges slightly
    beta_range = beta_max - beta_min
    m_range = m_max - m_min
    
    beta_grid = np.linspace(beta_min - 0.1 * beta_range, beta_max + 0.1 * beta_range, 100)
    m_grid = np.linspace(m_min - 0.1 * m_range, m_max + 0.1 * m_range, 100)
    
    beta_mesh, m_mesh = np.meshgrid(beta_grid, m_grid)
    
    # Interpolate ASR values
    try:
        asr_interpolated = griddata((betas, ms), asrs, (beta_mesh, m_mesh), method=interpolation_method)
    except Exception as e:
        print(f"⚠️  Interpolation failed with {interpolation_method}, falling back to 'nearest': {e}")
        asr_interpolated = griddata((betas, ms), asrs, (beta_mesh, m_mesh), method='nearest')
    
    # Create contour plot
    fig, ax = plt.subplots(figsize=figsize)
    
    # Create filled contour plot
    contour_filled = ax.contourf(beta_mesh, m_mesh, asr_interpolated, 
                                levels=n_levels, cmap='RdYlBu_r', alpha=0.8)
    
    # Add contour lines
    contour_lines = ax.contour(beta_mesh, m_mesh, asr_interpolated, 
                              levels=n_levels, colors='black', alpha=0.6, linewidths=0.8)
    
    # Add contour labels
    ax.clabel(contour_lines, inline=True, fontsize=10, fmt='%.1f%%')
    
    # Add colorbar
    cbar = plt.colorbar(contour_filled, ax=ax)
    cbar.set_label('Attack Success Rate (%)', fontsize=12)
    
    # Overlay actual data points
    scatter = ax.scatter(betas, ms, c=asrs, s=150, cmap='RdYlBu_r', 
                        edgecolors='black', linewidth=2, zorder=5)
    
    # Add data point labels
    for i, (beta, m, asr) in enumerate(zip(betas, ms, asrs)):
        ax.annotate(f'({beta}, {m})\n{asr:.1f}%', (beta, m), xytext=(10, 10), 
                   textcoords='offset points', fontsize=9, fontweight='bold',
                   bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.9))
    
    # Customize plot
    ax.set_xlabel('β (Beta Parameter)', fontsize=14)
    ax.set_ylabel('m (Caution num. tokens)', fontsize=14)
    # ax.set_title('Attack Success Rate Contour Plot\n(β vs m parameters)', fontsize=16)
    
    # Add grid
    ax.grid(True, alpha=0.3, linestyle='--')
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"💾 Parameter contour plot saved to: {output_path}")
    
    return fig, ax

def create_parameter_surface_3d(experiments_data, output_path=None, figsize=(12, 9)):
    """Create 3D surface plot of ASR vs beta and m parameters"""
    
    # Filter experiments with valid beta and m parameters
    param_experiments = {name: data for name, data in experiments_data.items() 
                        if data['beta'] is not None and data['m'] is not None}
    
    if len(param_experiments) < 3:
        print("❌ Need at least 3 experiments with beta/m parameters for 3D surface")
        return None, None
    
    # Extract data points
    betas = []
    ms = []
    asrs = []
    
    for exp_name, exp_data in param_experiments.items():
        betas.append(exp_data['beta'])
        ms.append(exp_data['m'])
        asrs.append(exp_data['categories']['Successful Attack']['percentage'])
    
    betas = np.array(betas)
    ms = np.array(ms)
    asrs = np.array(asrs)
    
    # Create 3D plot
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection='3d')
    
    # Create interpolation grid for surface
    beta_min, beta_max = betas.min(), betas.max()
    m_min, m_max = ms.min(), ms.max()
    
    beta_range = beta_max - beta_min
    m_range = m_max - m_min
    
    beta_grid = np.linspace(beta_min - 0.1 * beta_range, beta_max + 0.1 * beta_range, 30)
    m_grid = np.linspace(m_min - 0.1 * m_range, m_max + 0.1 * m_range, 30)
    
    beta_mesh, m_mesh = np.meshgrid(beta_grid, m_grid)
    
    # Interpolate for surface
    try:
        asr_surface = griddata((betas, ms), asrs, (beta_mesh, m_mesh), method='linear')
    except:
        asr_surface = griddata((betas, ms), asrs, (beta_mesh, m_mesh), method='nearest')
    
    # Create surface plot
    surf = ax.plot_surface(beta_mesh, m_mesh, asr_surface, cmap='RdYlBu_r', 
                          alpha=0.8, edgecolor='none')
    
    # Add actual data points
    ax.scatter(betas, ms, asrs, c=asrs, s=100, cmap='RdYlBu_r', 
              edgecolors='black', linewidth=2, zorder=5)
    
    # Customize plot
    ax.set_xlabel('β (Beta Parameter)', fontsize=12)
    ax.set_ylabel('m (Refusal Direction Dimension)', fontsize=12)
    ax.set_zlabel('Attack Success Rate (%)', fontsize=12)
    ax.set_title('3D Surface Plot: Attack Success Rate vs Parameters', fontsize=14)
    
    # Add colorbar
    fig.colorbar(surf, ax=ax, shrink=0.5, aspect=20, label='Attack Success Rate (%)')
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"💾 3D surface plot saved to: {output_path}")
    
    return fig, ax

def create_comparison_chart(experiments_data, output_path=None, 
                          title="Model Response Classification Comparison",
                          figsize=(18, 7), custom_spacing=None):
    """Create paper-quality comparison chart with stacked horizontal bars and custom spacing
    
    Args:
        custom_spacing: Dict specifying extra spacing after certain experiments
                       e.g., {"No jailbreak": 0.3, "Naïve GCG\n'</think>\\n\\nSure'": 0.5}
                       Values represent additional spacing units to add after that experiment
    """
    
    if not experiments_data:
        print("❌ No experiment data to plot")
        return None
    
    # Prepare data
    experiment_names = list(experiments_data.keys())
    n_experiments = len(experiment_names)
    
    # Category information with consistent colors
    category_info = {
        'Successful Attack': {'color': '#F12E2E', 'label': 'Successful Attack'},
        'Off-task': {'color': '#808080', 'label': 'Off-task'},
        'Refusal': {'color': "#407BF2", 'label': 'Unsuccessful Attack'}  # Changed label here
    }
    
    # Create figure with consistent styling
    fig, ax = plt.subplots(figsize=figsize)
    
    # Calculate positions with custom spacing
    y_positions = []
    current_pos = 0
    
    for i, exp_name in enumerate(experiment_names):
        y_positions.append(current_pos)
        current_pos += 1  # Base spacing of 1 unit
        
        # Add custom spacing if specified
        if custom_spacing and exp_name in custom_spacing:
            current_pos += custom_spacing[exp_name]
    
    y_positions = np.array(y_positions)
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
                       alpha=1.0)  # Full opacity like examples
                
                # Add count labels (if space allows)
                if percentage > 4:  # Only show if segment is wide enough
                    ax.text(cumulative + percentage/2, y_positions[i], 
                           f'{count}', ha='center', va='center', 
                           fontweight='normal', fontsize=14, color='white')
                
                cumulative += percentage
    
    # Customize chart to match example styling
    ax.set_xlim(0, 100)
    
    # Adjust y-axis limits to accommodate custom spacing
    y_min = min(y_positions) - 0.5
    y_max = max(y_positions) + 0.5
    ax.set_ylim(y_min, y_max)
    
    # Set y-axis
    ax.set_yticks(y_positions)
    ax.set_yticklabels(experiment_names, fontsize=16)
    ax.invert_yaxis()  # Top to bottom
    
    # Set x-axis
    ax.set_xlabel('Percentage of Responses (%)', fontsize=16)
    ax.set_xticks(range(0, 101, 20))
    ax.set_xticklabels([f"{x}" for x in range(0, 101, 20)])
    
    ax.tick_params(axis='both', which='major', labelsize=14)

    # Title with consistent styling (commented out as in original)
    # ax.set_title(title, fontsize=16)
    
    # Grid styling to match examples
    ax.grid(axis='x', linestyle='--', alpha=0.7)
    ax.set_axisbelow(True)
    
    # Create legend with consistent styling
    legend_elements = []
    for category_name, category_props in category_info.items():
        legend_elements.append(plt.Rectangle((0,0), 1, 1, 
                                           facecolor=category_props['color'], 
                                           alpha=1.0,
                                           label=category_props['label']))
    
    # Place legend centred above the graph
    ax.legend(handles=legend_elements, 
            bbox_to_anchor=(0.5, -0.18 ),  # Position above the plot, horizontally centered
            loc='center',                # Center the legend at that position
            fontsize=14, 
            title_fontsize=12,
            ncol=3)  
    
    plt.tight_layout()
    
    # Save if path provided
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"💾 Comparison chart saved to: {output_path}")
    
    return fig, ax

def create_success_rate_comparison(experiments_data, output_path=None):
    """Create a focused comparison of success rates with consistent styling"""
    experiment_names = list(experiments_data.keys())
    success_rates = []
    
    for exp_data in experiments_data.values():
        success_rates.append(exp_data['categories']['Successful Attack']['percentage'])
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Create bar chart with consistent colors
    bars = ax.bar(range(len(experiment_names)), success_rates, 
                  color='#ff9900', alpha=1.0)  # Use orange from examples
    
    # Add value labels on bars
    for i, (bar, rate) in enumerate(zip(bars, success_rates)):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                f'{rate:.1f}%', ha='center', va='bottom', fontweight='normal')
    
    # Customize to match example styling
    ax.set_xlabel('Experiment', fontsize=12)
    ax.set_ylabel('Attack Success Rate (%)', fontsize=12)
    ax.set_title('Attack Success Rate Comparison', fontsize=16)
    ax.set_xticks(range(len(experiment_names)))
    ax.set_xticklabels(experiment_names, rotation=45, ha='right')
    ax.set_ylim(0, max(success_rates) * 1.1 + 5)
    
    # Grid styling to match examples
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    ax.set_axisbelow(True)
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"💾 Success rate comparison saved to: {output_path}")
    
    return fig, ax

def create_summary_table(experiments_data):
    """Create a summary table of results"""
    print("\n" + "="*80)
    print("📊 MULTI-EXPERIMENT COMPARISON SUMMARY")
    print("="*80)
    
    # Headers
    print(f"{'Experiment':.<25} {'Total':>8} {'Refusal':>10} {'Off-task':>10} {'Success':>10} {'Success %':>10} {'β':>6} {'m':>4}")
    print("-" * 80)
    
    for exp_name, exp_data in experiments_data.items():
        categories = exp_data['categories']
        total = exp_data['total_ranked']
        
        refusal_count = categories['Refusal']['count']
        offtask_count = categories['Off-task']['count']
        success_count = categories['Successful Attack']['count']
        success_pct = categories['Successful Attack']['percentage']
        
        beta = exp_data.get('beta', 'N/A')
        m = exp_data.get('m', 'N/A')
        
        beta_str = f"{beta}" if beta is not None else "N/A"
        m_str = f"{m}" if m is not None else "N/A"
        
        print(f"{exp_name:.<25} {total:>8} {refusal_count:>10} {offtask_count:>10} "
              f"{success_count:>10} {success_pct:>9.1f}% {beta_str:>6} {m_str:>4}")
    
    print("="*80)

def create_detailed_boxplot_comparison(experiments_data, output_path=None):
    """Create boxplot comparison similar to the examples provided"""
    
    # Prepare data for boxplot
    all_success_rates = []
    experiment_labels = []
    
    for exp_name, exp_data in experiments_data.items():
        success_rate = exp_data['categories']['Successful Attack']['percentage']
        all_success_rates.append(success_rate)
        experiment_labels.append(exp_name)
    
    # Create DataFrame for seaborn
    df_box = pd.DataFrame({
        'Success Rate': all_success_rates,
        'Experiment': experiment_labels
    })
    
    # Create figure
    plt.figure(figsize=(12, 8))
    
    # Create boxplot with consistent styling
    ax = sns.boxplot(x='Success Rate', y='Experiment', data=df_box, 
                    palette=['#ff9900'] * len(experiment_labels),
                    width=0.5, showmeans=True, 
                    meanprops={"marker":"o", "markerfacecolor":"white", 
                             "markeredgecolor":"black", "markersize":"8"})
    
    # Add individual data points
    sns.stripplot(x='Success Rate', y='Experiment', data=df_box, 
                 color='#805000', size=6, alpha=0.7, jitter=True)
    
    # Add grid and style to match examples
    plt.grid(axis='x', linestyle='--', alpha=0.7)
    plt.xlim(0, 100)
    
    # Add title and labels
    # plt.title('Distribution of Attack Success Rates Across Experiments', fontsize=16)
    plt.xlabel('Attack Success Rate (%)', fontsize=12)
    plt.ylabel('', fontsize=12)
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"💾 Boxplot comparison saved to: {output_path}")
    
    plt.show()

def main_comparison(experiment_dict, output_dir="./", base_filename="experiment_comparison", 
                   bar_spacing=None, create_parameter_plots=True):
    """Main function for multi-experiment comparison with parameter plots
    
    Args:
        bar_spacing: Dict specifying extra spacing after certain experiments
        create_parameter_plots: Whether to create heatmap and contour plots
    """
    
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
    
    # Generate standard visualizations
    comparison_chart_path = output_path / f"{base_filename}_comparison.png"
    success_rate_chart_path = output_path / f"{base_filename}_success_rates.png"
    boxplot_path = output_path / f"{base_filename}_boxplot.png"
    
    print(f"\n📈 Creating comparison chart...")
    fig1, ax1 = create_comparison_chart(experiments_data, comparison_chart_path, 
                                       custom_spacing=bar_spacing)
    
    print(f"📊 Creating success rate comparison...")
    fig2, ax2 = create_success_rate_comparison(experiments_data, success_rate_chart_path)
    
    print(f"📦 Creating boxplot comparison...")
    create_detailed_boxplot_comparison(experiments_data, boxplot_path)
    
    # Generate parameter plots if requested
    if create_parameter_plots:
        # Check if we have parameter experiments
        param_experiments = {name: data for name, data in experiments_data.items() 
                           if data.get('beta') is not None and data.get('m') is not None}
        
        if len(param_experiments) >= 3:
            print(f"\n🎯 Creating parameter analysis plots...")
            
            heatmap_path = output_path / f"{base_filename}_parameter_heatmap.png"
            contour_path = output_path / f"{base_filename}_parameter_contour.png"
            surface_path = output_path / f"{base_filename}_parameter_surface_3d.png"
            
            print(f"🔥 Creating parameter heatmap...")
            fig3, ax3 = create_parameter_heatmap(experiments_data, heatmap_path)
            
            print(f"📈 Creating parameter contour plot...")
            fig4, ax4 = create_parameter_contour(experiments_data, contour_path)
            
            print(f"🏔️  Creating 3D surface plot...")
            fig5, ax5 = create_parameter_surface_3d(experiments_data, surface_path)
        else:
            print(f"⚠️  Only {len(param_experiments)} experiments with β/m parameters found.")
            print("   Need at least 3 for meaningful parameter plots.")
    
    # Show plots
    plt.show()
    
    print(f"\n✅ Multi-experiment analysis complete!")
    print(f"📁 Files saved to: {output_path}")
    
    return experiments_data

def filter_parameter_experiments(experiment_dict):
    """Filter experiment dict to only include those with beta/m parameters
    Excludes baseline experiments like 'No jailbreak' and 'Naïve GCG'
    """
    filtered_experiments = OrderedDict()
    
    for exp_name, csv_path in experiment_dict.items():
        # Skip baseline experiments that don't have beta/m parameters
        if any(skip_term in exp_name.lower() for skip_term in ['no jailbreak', 'naïve gcg', 'naive gcg']):
            continue
            
        # Only include experiments with "Force orthogonalised" or clear beta/m indicators
        if "Force orthogonalised" in exp_name or any(param in exp_name for param in ['β=', 'm=']):
            filtered_experiments[exp_name] = csv_path
    
    return filtered_experiments

def create_parameter_only_analysis(experiment_dict, output_dir="./", base_filename="parameter_analysis"):
    """Run analysis focusing only on experiments with beta/m parameters"""
    
    print("🎯 Running parameter-focused analysis...")
    
    # Filter to parameter experiments only
    param_experiments = filter_parameter_experiments(experiment_dict)
    
    if len(param_experiments) < 3:
        print("❌ Not enough parameter experiments found for analysis")
        print(f"   Found: {list(param_experiments.keys())}")
        return None
    
    print(f"✅ Found {len(param_experiments)} parameter experiments:")
    for exp_name in param_experiments.keys():
        print(f"   - {exp_name}")
    
    # Run main comparison with parameter plots enabled
    return main_comparison(param_experiments, output_dir, base_filename, 
                          create_parameter_plots=True)

# Example usage
if __name__ == "__main__":
    # Define your experiments - modify these paths as needed
    experiments = OrderedDict([
        ("No jailbreak\n5 repeat rollouts", "rankings/base_5rollouts.csv"),
        ("Naïve GCG\n'</think>\\n\\nSure'", "rankings/naive_gcg.csv"),
        ("Force orthogonalised output only\n(β=0)", "rankings/force_ortho.csv"),
        ("Force orthogonalised output + \nCaution minimisation with β=0.3, m=45", "rankings/beta0.3.csv"),
        ("Force orthogonalised output + \nCaution minimisation with β=0.5, m=45", "rankings/beta0.5.csv"),
        ("Force orthogonalised output + \nCaution minimisation with β=0.7, m=45", "rankings/beta0.7.csv"),
        ("Force orthogonalised output + \nCaution minimisation with β=0.9, m=45", "rankings/beta0.9.csv"),
        ("Force orthogonalised output + \nCaution minimisation with β=0.5, m=70", "rankings/beta0.5_refusal70.csv"),
        ("Force orthogonalised output + \nCaution minimisation with β=0.7, m=70", "rankings/beta0.7_refusal70.csv")
    ])
    # Define custom spacing between specific bars
    # Add extra spacing after baseline and naive GCG to separate method groups
    custom_bar_spacing = {
        # "No jailbreak": 0.4,  # Add extra space after baseline
        "Naïve GCG\n'</think>\\n\\nSure'": 0.4,  # Add larger space after naive GCG
        # "Force orthogonalised output only\n(β=0)": 0.4,  # Small space after beta=0
        "Force orthogonalised output + \nCaution minimisation with β=0.9, m=45": 0.4
    }



    
    
    
    # Option 1: Run full comparison including baseline experiments
    print("="*60)
    print("FULL EXPERIMENT COMPARISON")
    print("="*60)
    main_comparison(experiments, output_dir="./figures/", 
                   base_filename="gcg_experiments", 
                   bar_spacing=custom_bar_spacing,
                   create_parameter_plots=True)
    
    # Option 2: Run parameter-only analysis (excluding baseline experiments)
    print("\n" + "="*60)
    print("PARAMETER-ONLY ANALYSIS")
    print("="*60)
    create_parameter_only_analysis(experiments, output_dir="./figures/", 
                                  base_filename="parameter_analysis")
    
    print("\n🎉 Analysis complete! Check the ./figures/ directory for all outputs.")
    
    # Example of how to add new experiments following the naming schema:
    print("\n" + "="*60)
    print("ADDING NEW EXPERIMENTS")
    print("="*60)
    print("To add new experiments, follow this naming pattern:")
    print("- 'Force orthogonalised output only (β=0)' for β=0, m=0")
    print("- 'Force orthogonalised output + Caution minimisation with β=X.X, m=YY' for other values")
    print("\nExample new experiments:")
    print("- 'Force orthogonalised output + Caution minimisation with β=0.5, m=30'")
    print("- 'Force orthogonalised output + Caution minimisation with β=1.0, m=60'")
    print("\nThe parameter extraction will automatically work with this naming scheme!")
    
    # Show parameter extraction for existing experiments
    print("\n📊 Current parameter extraction:")
    for exp_name in experiments.keys():
        beta, m = extract_beta_m_parameters(exp_name)
        if beta is not None and m is not None:
            print(f"  {exp_name[:50]}... → β={beta}, m={m}")
        else:
            print(f"  {exp_name[:50]}... → No parameters (baseline)")

    print("Graphs for transfer attack success")
    transfer_experiments = OrderedDict([
        # ("Original jailbreak performance\n(β=0.5, m=70')", "rankings/beta0.5_refusal70.csv"),
        ("DeepSeek-R1-Distill-Qwen-7B Baseline", "transfer/qwen7b_baseline_ranked.csv"),
        ("Transfer to\nDeepSeek-R1-Distill-Qwen-7Bs", "transfer/qwen7b_ranked.csv"),
        ("DeepSeek-R1-Distill-Qwen-14B Baseline", "transfer/qwen14b_baseline_ranked.csv"),
        ("Transfer to\nDeepSeek-R1-Distill-Qwen-14B", "transfer/qwen14b_ranked.csv")
    ])
    transfer_data = load_multiple_experiments(transfer_experiments)
    transfer_spacing = {
        "Transfer to\nDeepSeek-R1-Distill-Qwen-7B": 0.4,  # Add space after original
    }

    # make bar comparison chart for transfer experiments
    create_comparison_chart(transfer_data,custom_spacing=transfer_spacing, output_path="./figures/transfer_comparison.png", figsize=(10,6))