import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from carbon_optimizer import CarbonOptimizer, CarbonParameters
import json

# Page config
st.set_page_config(
    page_title="Azalt: Carbon Risk Co Pilot - Batıçim",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Enterprise-grade styling
st.markdown("""
<style>
    /* Main container */
    .main > div {
        padding-top: 1rem;
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    }
    
    /* Header styling */
    .main-header {
        background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%);
        color: white;
        padding: 2rem;
        border-radius: 10px;
        margin-bottom: 2rem;
        text-align: center;
        box-shadow: 0 4px 20px rgba(0,0,0,0.1);
    }
    
    /* Metrics styling */
    .stMetric {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
        border: none;
        padding: 1.5rem;
        border-radius: 12px;
        margin: 0.5rem 0;
        box-shadow: 0 2px 10px rgba(0,0,0,0.08);
        border-left: 4px solid #2a5298;
    }
    
    /* Sidebar styling */
    .css-1d391kg {
        background-color: #f8f9fa;
        border-right: 2px solid #e9ecef;
    }
    
    /* Chart containers */
    .chart-container {
        background: white;
        border-radius: 12px;
        padding: 1rem;
        margin: 1rem 0;
        box-shadow: 0 2px 15px rgba(0,0,0,0.05);
        border: 1px solid #e9ecef;
    }
    
    /* Subheaders */
    .subheader {
        color: #2c3e50;
        font-weight: 600;
        margin: 1.5rem 0 1rem 0;
        padding-bottom: 0.5rem;
        border-bottom: 2px solid #3498db;
    }
    
    /* Data table styling */
    .dataframe {
        border: none !important;
        border-radius: 8px;
        overflow: hidden;
        box-shadow: 0 2px 10px rgba(0,0,0,0.05);
    }
    
    /* Button styling */
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 0.75rem 1.5rem;
        font-weight: 600;
        transition: all 0.3s ease;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 15px rgba(0,0,0,0.2);
    }
    
    /* Sidebar parameter styling */
    .sidebar-section {
        background: white;
        border-radius: 8px;
        padding: 1rem;
        margin: 1rem 0;
        border-left: 4px solid #3498db;
    }
</style>
""", unsafe_allow_html=True)

def main():
    # Main header with clean, modest styling
    st.markdown("""
    <div style="text-align: center; margin: 2rem 0; padding: 1rem 0;">
        <h1 style="color: #2c3e50; font-family: 'Segoe UI', sans-serif; font-weight: 300; font-size: 2.5rem; margin: 0; letter-spacing: 2px;">
            Azalt: Carbon Risk Co Pilot - Batıçim
        </h1>
        <p style="color: #7f8c8d; font-size: 1.1rem; margin: 0.5rem 0 0 0; font-weight: 400;">
            Enterprise-grade carbon cost modeling for 2024-2034
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar for parameters
    st.sidebar.header("PLANT PARAMETERS")
    
    # Core production parameters
    with st.sidebar.expander("PRODUCTION PARAMETERS", expanded=True):
        plant_capacity = st.number_input(
            "Plant Capacity (t cement/yr)", 
            min_value=500_000, 
            max_value=3_000_000, 
            value=1_759_584,
            step=50_000,
            help="Total annual cement production capacity"
        )
        
        clinker_ratio = st.slider(
            "Clinker-to-Cement Ratio", 
            min_value=0.60, 
            max_value=0.90, 
            value=0.80,
            step=0.01,
            help="Proportion of clinker in cement (higher = more CO2)"
        )
        
        eu_export_volume = st.number_input(
            "EU Export Volume (t/yr)", 
            min_value=0, 
            max_value=int(plant_capacity), 
            value=min(650_000, int(plant_capacity)),
            step=25_000,
            help="Annual cement exports to EU (subject to CBAM)"
        )
    
    # Emissions parameters
    with st.sidebar.expander("EMISSIONS PARAMETERS", expanded=True):
        co2_process_intensity = st.slider(
            "CO2 Process Intensity (kg CO2/t clinker)", 
            min_value=700, 
            max_value=950, 
            value=904,
            step=5,
            help="CO2 emissions from limestone calcination"
        )
        
        thermal_energy = st.slider(
            "Thermal Energy (MJ/t clinker)", 
            min_value=3000, 
            max_value=4500, 
            value=3830,
            step=25,
            help="Thermal energy consumption per tonne clinker"
        )
        
        alt_fuel_share = st.slider(
            "Alternative Fuel Share (%)", 
            min_value=0, 
            max_value=50, 
            value=0,
            step=1,
            help="Percentage of thermal energy from alternative fuels"
        ) / 100
        
        renewable_el_share = st.slider(
            "Renewable Electricity Share (%)", 
            min_value=0, 
            max_value=100, 
            value=30,
            step=1,
            help="Percentage of electricity from renewable sources"
        ) / 100
        
        electricity_use = st.slider(
            "Electricity Use (kWh/t cement)", 
            min_value=50, 
            max_value=150, 
            value=78,
            step=2,
            help="Electricity consumption per tonne cement"
        )
    
    # Carbon pricing parameters
    with st.sidebar.expander("CARBON PRICING", expanded=True):
        cbam_price_2026 = st.slider(
            "CBAM Price 2026 (USD/t CO2)", 
            min_value=50, 
            max_value=200, 
            value=90,
            step=5,
            help="CBAM certificate price in 2026"
        )
        
        tr_ets_price_2026 = st.slider(
            "TR-ETS Price 2026 (USD/t CO2)", 
            min_value=5, 
            max_value=50, 
            value=9,
            step=1,
            help="Turkey ETS allowance price in 2026"
        )
    
    # Free allocation parameters
    with st.sidebar.expander("FREE ALLOCATION", expanded=True):
        eu_benchmark = st.slider(
            "EU Benchmark (t CO2/t cement)", 
            min_value=0.500, 
            max_value=0.900, 
            value=0.693,
            step=0.005,
            help="EU-ETS Phase 4 benchmark for cement"
        )
        
        tr_ets_free_share = st.slider(
            "TR-ETS Free Allocation Share (%)", 
            min_value=0, 
            max_value=100, 
            value=100,
            step=1,
            help="Percentage of TR-ETS allowances allocated for free"
        ) / 100
    
    # Create parameters object
    params = CarbonParameters(
        plant_capacity_tpa=plant_capacity,
        co2_process_intensity=co2_process_intensity,
        clinker_ratio=clinker_ratio,
        thermal_energy_MJ=thermal_energy,
        alt_fuel_share=alt_fuel_share,
        renewable_el_share=renewable_el_share,
        electricity_use_kWh=electricity_use,
        eu_export_volume=eu_export_volume,
        cbam_price_usd_2026=cbam_price_2026,
        tr_ets_price_usd_2026=tr_ets_price_2026,
        eu_benchmark_t_co2_per_t_cement=eu_benchmark,
        tr_ets_free_allocation_share=tr_ets_free_share
    )
    
    # Run optimization
    optimizer = CarbonOptimizer(params)
    results = optimizer.run_optimization()
    
    # Key Performance Indicators
    st.markdown('<div class="subheader">KEY PERFORMANCE INDICATORS</div>', unsafe_allow_html=True)
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            "Total 11-Year Cost",
            f"${results['Total_Cost_USD'].sum():,.0f}",
            help="Total carbon-related costs 2024-2034"
        )
    
    with col2:
        st.metric(
            "Average Annual Cost",
            f"${results['Total_Cost_USD'].mean():,.0f}",
            help="Average annual carbon cost"
        )
    
    with col3:
        st.metric(
            "Peak Annual Cost",
            f"${results['Total_Cost_USD'].max():,.0f}",
            help=f"Maximum annual cost (Year {results.loc[results['Total_Cost_USD'].idxmax(), 'Year']})"
        )
    
    with col4:
        st.metric(
            "Cost per Tonne",
            f"${results['Cost_per_Tonne_USD'].mean():.2f}",
            help="Average cost per tonne cement produced"
        )
    
    # Cost breakdown chart
    st.markdown('<div class="subheader">ANNUAL COST BREAKDOWN</div>', unsafe_allow_html=True)
    
    fig_breakdown = go.Figure()
    
    fig_breakdown.add_trace(go.Scatter(
        x=results['Year'],
        y=results['CBAM_Cost_USD'],
        mode='lines+markers',
        name='CBAM Cost',
        line=dict(color='#e74c3c', width=3)
    ))
    
    fig_breakdown.add_trace(go.Scatter(
        x=results['Year'],
        y=results['TR_ETS_Cost_USD'],
        mode='lines+markers',
        name='TR-ETS Cost',
        line=dict(color='#3498db', width=3)
    ))
    
    fig_breakdown.add_trace(go.Scatter(
        x=results['Year'],
        y=results['Freight_Cost_USD'],
        mode='lines+markers',
        name='Freight ETS Cost',
        line=dict(color='#2ecc71', width=3)
    ))
    
    fig_breakdown.add_trace(go.Scatter(
        x=results['Year'],
        y=results['Total_Cost_USD'],
        mode='lines+markers',
        name='Total Cost',
        line=dict(color='#2c3e50', width=4),
        yaxis='y2'
    ))
    
    fig_breakdown.update_layout(
        title=dict(
            text="Carbon Cost Evolution 2024-2034",
            font=dict(size=20, family="Arial, sans-serif", color="#2c3e50"),
            x=0.5
        ),
        xaxis_title="Year",
        yaxis_title="Individual Cost Components (USD)",
        yaxis2=dict(
            title="Total Cost (USD)",
            overlaying='y',
            side='right'
        ),
        hovermode='x unified',
        height=500,
        showlegend=True,
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(family="Arial, sans-serif", color="#2c3e50"),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="center",
            x=0.5
        )
    )
    
    st.plotly_chart(fig_breakdown, use_container_width=True)
    
    # Free allocation impact
    st.markdown('<div class="subheader">FREE ALLOCATION IMPACT</div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        fig_cbam = go.Figure()
        fig_cbam.add_trace(go.Scatter(
            x=results['Year'],
            y=results['CBAM_Phase_In_Factor'],
            mode='lines+markers',
            name='CBAM Phase-In Factor',
            line=dict(color='#FF6B6B', width=3),
            fill='tozeroy'
        ))
        fig_cbam.update_layout(
            title="CBAM Phase-In Schedule",
            xaxis_title="Year",
            yaxis_title="Phase-In Factor",
            yaxis=dict(range=[0, 1.1]),
            height=400
        )
        st.plotly_chart(fig_cbam, use_container_width=True)
    
    with col2:
        fig_tr_ets = go.Figure()
        fig_tr_ets.add_trace(go.Scatter(
            x=results['Year'],
            y=results['TR_ETS_Free_Alloc_t'],
            mode='lines+markers',
            name='TR-ETS Free Allowances',
            line=dict(color='#4ECDC4', width=3),
            fill='tozeroy'
        ))
        fig_tr_ets.update_layout(
            title="TR-ETS Free Allowances (Annual)",
            xaxis_title="Year",
            yaxis_title="Free Allowances (t CO2)",
            height=400
        )
        st.plotly_chart(fig_tr_ets, use_container_width=True)
    
    # Sensitivity analysis
    st.markdown('<div class="subheader">SENSITIVITY ANALYSIS</div>', unsafe_allow_html=True)
    
    sensitivity = optimizer.sensitivity_analysis(results)
    
    # Create tornado chart
    fig_tornado = go.Figure()
    
    # Sort by absolute range for better visualization
    sensitivity_sorted = sensitivity.sort_values('Range_USD', key=abs, ascending=True)
    
    fig_tornado.add_trace(go.Bar(
        y=sensitivity_sorted['Parameter'],
        x=sensitivity_sorted['Low_Impact_USD'],
        orientation='h',
        name='Low Impact (-10%)',
        marker_color='#FF6B6B',
        text=[f"${x:,.0f}" for x in sensitivity_sorted['Low_Impact_USD']],
        textposition='inside'
    ))
    
    fig_tornado.add_trace(go.Bar(
        y=sensitivity_sorted['Parameter'],
        x=sensitivity_sorted['High_Impact_USD'],
        orientation='h',
        name='High Impact (+10%)',
        marker_color='#4ECDC4',
        text=[f"${x:,.0f}" for x in sensitivity_sorted['High_Impact_USD']],
        textposition='inside'
    ))
    
    fig_tornado.update_layout(
        title="Sensitivity Tornado Chart (±10% Parameter Changes)",
        xaxis_title="Impact on Total Cost (USD)",
        yaxis_title="Parameter",
        barmode='relative',
        height=600,
        showlegend=True
    )
    
    st.plotly_chart(fig_tornado, use_container_width=True)
    
    # Data table
    st.markdown('<div class="subheader">DETAILED RESULTS</div>', unsafe_allow_html=True)
    
    # Format the results for better display
    display_results = results.copy()
    display_results['Total_Cost_USD'] = display_results['Total_Cost_USD'].apply(lambda x: f"${x:,.0f}")
    display_results['CBAM_Cost_USD'] = display_results['CBAM_Cost_USD'].apply(lambda x: f"${x:,.0f}")
    display_results['TR_ETS_Cost_USD'] = display_results['TR_ETS_Cost_USD'].apply(lambda x: f"${x:,.0f}")
    display_results['Freight_Cost_USD'] = display_results['Freight_Cost_USD'].apply(lambda x: f"${x:,.0f}")
    display_results['Cost_per_Tonne_USD'] = display_results['Cost_per_Tonne_USD'].apply(lambda x: f"${x:.2f}")
    display_results['Total_CO2_t'] = display_results['Total_CO2_t'].apply(lambda x: f"{x:,.0f}")
    display_results['CBAM_Phase_In_Factor'] = display_results['CBAM_Phase_In_Factor'].apply(lambda x: f"{x:.1%}")
    display_results['TR_ETS_Free_Alloc_t'] = display_results['TR_ETS_Free_Alloc_t'].apply(lambda x: f"{x:,.0f}")
    display_results['Freight_Phase_In_Factor'] = display_results['Freight_Phase_In_Factor'].apply(lambda x: f"{x:.1%}")
    
    st.dataframe(display_results, use_container_width=True, height=400)
    
    # Download results
    st.markdown('<div class="subheader">EXPORT RESULTS</div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        csv = results.to_csv(index=False)
        st.download_button(
            label="Download Results CSV",
            data=csv,
            file_name="carbon_cost_projections.csv",
            mime="text/csv"
        )
    
    with col2:
        sensitivity_csv = sensitivity.to_csv(index=False)
        st.download_button(
            label="Download Sensitivity CSV",
            data=sensitivity_csv,
            file_name="sensitivity_analysis.csv",
            mime="text/csv"
        )
    
    # Assumptions section
    st.markdown('<div class="subheader">KEY ASSUMPTIONS</div>', unsafe_allow_html=True)
    
    assumptions_text = """
    **CARBON PRICING ASSUMPTIONS:**
    • CBAM certificate prices follow EU ETS trajectory with 5-7% annual growth
    • TR-ETS prices start at $9/tCO2 in 2026 with gradual convergence toward EU prices
    • Freight ETS started in 2024 at 40%, increased to 70% in 2025, full implementation from 2026
    
    **PRODUCTION ASSUMPTIONS:**
    • Plant capacity utilization remains constant at current levels
    • Clinker-to-cement ratio reflects current production mix
    • EU export volumes maintain current market penetration
    
    **EMISSIONS ASSUMPTIONS:**
    • Process emissions based on limestone calcination stoichiometry
    • Thermal energy requirements reflect current kiln efficiency
    • Electricity grid carbon intensity follows national energy transition plans
    
    **REGULATORY ASSUMPTIONS:**
    • CBAM implementation follows EU timeline (2026 full implementation)
    • TR-ETS free allocation declines 2.2% annually from 2026
    • EU benchmark values remain constant at Phase 4 levels
    
    **FINANCIAL ASSUMPTIONS:**
    • USD exchange rates remain stable relative to EUR and TRY
    • No additional carbon taxes or levies beyond modeled systems
    • Transportation costs exclude potential modal shift impacts
    """
    
    st.markdown(assumptions_text)
    
    # Footer
    st.markdown("---")
    st.markdown(
        "**AZALT: CARBON RISK CO PILOT - BATIÇIM** | "
        "Enterprise Carbon Analytics Platform | "
        "Comprehensive CBAM, TR-ETS, and Freight ETS modeling with free allocation mechanisms"
    )

if __name__ == "__main__":
    main()