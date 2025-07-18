#!/usr/bin/env python3
"""
Batıçim Cement — 2025-2034 Carbon-Cost Optimizer (v0)
Decision-support tool for minimizing carbon-related cash-out (USD only)
"""

import json
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Any
from dataclasses import dataclass
import matplotlib.pyplot as plt
import seaborn as sns

@dataclass
class CarbonParameters:
    """Core carbon cost parameters (all USD) - Updated with Batıçim actuals"""
    plant_capacity_tpa: float = 1_759_584  # Actual Batıçim capacity
    co2_process_intensity: float = 904  # kg CO2/t clinker (2024 actual)
    clinker_ratio: float = 0.80  # 80% clinker in cement
    thermal_energy_MJ: float = 3_830  # MJ/t clinker (actual)
    alt_fuel_share: float = 0.0  # Currently 0% alternative fuels
    renewable_el_share: float = 0.30  # 30% renewable electricity
    grid_emission_factor: float = 0.442  # kg CO2e/kWh (Turkey grid)
    electricity_use_kWh: float = 77.64  # kWh/t cement (actual)
    eu_export_volume: float = 650_000  # t/yr actual exports
    cbam_price_usd_2026: float = 90  # USD/t CO2 (escalates 5%/yr)
    tr_ets_price_usd_2026: float = 9  # USD/t CO2 (starts 2026, escalates 5%/yr)
    freight_ets_pass_through_usd_2026: float = 1.00  # USD/t cement (escalates 3%/yr)
    
    # Free allocation parameters
    eu_benchmark_t_co2_per_t_cement: float = 0.693  # EU-ETS Phase 4 benchmark
    tr_ets_free_allocation_share: float = 1.0  # 100% free allocation initially
    cbam_free_allocation_phaseout_start: int = 2026
    cbam_free_allocation_phaseout_end: int = 2034
    
    # CBAM phase-in schedule (actual percentages)
    cbam_phase_in_schedule: Dict[int, float] = None
    
    # TR-ETS free allocation phase-down schedule (realistic assumption)
    tr_ets_free_allocation_schedule: Dict[int, float] = None
    
    # Freight ETS phase-in schedule (actual implementation)
    freight_ets_phase_in_schedule: Dict[int, float] = None
    
    escalators: Dict[str, float] = None
    
    def __post_init__(self):
        if self.escalators is None:
            self.escalators = {
                'cbam_price_usd': 0.05,  # 5%/yr
                'tr_ets_price_usd': 0.05,  # 5%/yr
                'freight_ets_pass_through_usd': 0.03  # 3%/yr
            }
        
        if self.cbam_phase_in_schedule is None:
            self.cbam_phase_in_schedule = {
                2026: 0.025,   # 2.5%
                2027: 0.050,   # 5.0%
                2028: 0.100,   # 10%
                2029: 0.225,   # 22.5%
                2030: 0.485,   # 48.5%
                2031: 0.610,   # 61%
                2032: 0.735,   # 73.5%
                2033: 0.860,   # 86%
                2034: 1.000    # 100%
            }
        
        if self.freight_ets_phase_in_schedule is None:
            self.freight_ets_phase_in_schedule = {
                2024: 0.40,    # 40% (actual implementation)
                2025: 0.70,    # 70% (actual implementation)
                # 100% from 2026 onwards (default 1.0)
            }

class CarbonOptimizer:
    """Main carbon cost optimization engine"""
    
    def __init__(self, params: CarbonParameters):
        self.params = params
        self.years = list(range(2024, 2035))
        self.results = {}
        
    def calculate_co2_emissions(self, year: int) -> Dict[str, float]:
        """Calculate total CO2 emissions by source"""
        
        # Process emissions (limestone calcination)
        process_co2 = (
            self.params.plant_capacity_tpa * 
            self.params.clinker_ratio * 
            self.params.co2_process_intensity / 1000  # t CO2
        )
        
        # Fuel emissions (thermal energy)
        fuel_co2_factor = 0.074  # kg CO2/MJ for coal baseline
        fuel_co2_reduction = self.params.alt_fuel_share * 0.85  # 85% reduction from alt fuels
        fuel_co2 = (
            self.params.plant_capacity_tpa * 
            self.params.clinker_ratio * 
            self.params.thermal_energy_MJ * 
            fuel_co2_factor * 
            (1 - fuel_co2_reduction) / 1000  # t CO2
        )
        
        # Electricity emissions (indirect)
        grid_electricity = self.params.electricity_use_kWh * (1 - self.params.renewable_el_share)
        electricity_co2 = (
            self.params.plant_capacity_tpa * 
            grid_electricity * 
            self.params.grid_emission_factor / 1000  # t CO2
        )
        
        return {
            'process_co2': process_co2,
            'fuel_co2': fuel_co2,
            'electricity_co2': electricity_co2,
            'total_co2': process_co2 + fuel_co2 + electricity_co2
        }
    
    def get_escalated_price(self, base_price: float, escalator_key: str, base_year: int, target_year: int) -> float:
        """Apply escalation to prices"""
        escalator = self.params.escalators.get(escalator_key, 0.0)
        years_diff = target_year - base_year
        return base_price * (1 + escalator) ** years_diff
    
    def calculate_cbam_phase_in_factor(self, year: int) -> float:
        """Calculate CBAM phase-in factor (actual schedule)"""
        if year < 2026:
            return 0.0  # No CBAM before 2026
        elif year > 2034:
            return 1.0  # Full CBAM after 2034
        else:
            # Use actual phase-in schedule
            return self.params.cbam_phase_in_schedule.get(year, 1.0)
    
    def calculate_cbam_cost(self, year: int, emissions: Dict[str, float]) -> Dict[str, float]:
        """Calculate CBAM costs for EU exports with actual phase-in schedule (USD)"""
        if year < 2026:
            return {'cbam_cost_usd': 0.0, 'cbam_phase_in_factor': 0.0}
        
        # CBAM covers process + fuel + electricity emissions
        cbam_emissions = emissions['total_co2']
        
        # Only applies to EU exports
        export_share = self.params.eu_export_volume / self.params.plant_capacity_tpa
        cbam_emissions_export = cbam_emissions * export_share
        
        # Calculate CBAM phase-in factor (actual schedule)
        phase_in_factor = self.calculate_cbam_phase_in_factor(year)
        
        # Apply TR-ETS carbon pricing credit (reduces CBAM obligation)
        tr_ets_credit_factor = 0.0
        if year >= 2026:  # TR-ETS starts in 2026
            tr_ets_price_usd = self.get_escalated_price(
                self.params.tr_ets_price_usd_2026, 'tr_ets_price_usd', 2026, year
            )
            cbam_price_usd = self.get_escalated_price(
                self.params.cbam_price_usd_2026, 'cbam_price_usd', 2026, year
            )
            # Credit = min(TR-ETS price / CBAM price, 1.0)
            tr_ets_credit_factor = min(tr_ets_price_usd / cbam_price_usd, 1.0)
        
        # Net CBAM obligation after phase-in and TR-ETS credit
        net_cbam_factor = phase_in_factor * (1 - tr_ets_credit_factor)
        net_cbam_emissions = cbam_emissions_export * net_cbam_factor
        
        # Get escalated CBAM price
        cbam_price_usd = self.get_escalated_price(
            self.params.cbam_price_usd_2026, 'cbam_price_usd', 2026, year
        )
        
        cbam_cost_usd = net_cbam_emissions * cbam_price_usd
        
        return {
            'cbam_cost_usd': cbam_cost_usd,
            'cbam_price_usd': cbam_price_usd,
            'cbam_emissions_t': cbam_emissions_export,
            'cbam_phase_in_factor': phase_in_factor,
            'cbam_tr_ets_credit_factor': tr_ets_credit_factor,
            'cbam_net_emissions_t': net_cbam_emissions
        }
    
    def calculate_tr_ets_free_allocation(self, year: int) -> float:
        """Calculate TR-ETS free allocation - mirrors EU-ETS (inverse of CBAM phase-in)"""
        if year < 2026:
            return 0.0
            
        # Free allocation = Benchmark × Production × (1 - CBAM phase-in rate)
        cement_production = self.params.plant_capacity_tpa
        
        # Convert benchmark from t CO2/t cement to total allowances
        benchmark_allowances = cement_production * self.params.eu_benchmark_t_co2_per_t_cement
        
        # TR-ETS free allocation = 1 - CBAM phase-in rate
        cbam_phase_in_rate = self.calculate_cbam_phase_in_factor(year)
        tr_ets_free_allocation_rate = 1.0 - cbam_phase_in_rate
        
        # Apply free allocation rate
        free_allocation = benchmark_allowances * tr_ets_free_allocation_rate
        
        return free_allocation
    
    def calculate_tr_ets_cost(self, year: int, emissions: Dict[str, float]) -> Dict[str, float]:
        """Calculate TR-ETS costs for domestic production with free allocation (USD)"""
        if year < 2026:
            return {'tr_ets_cost_usd': 0.0, 'tr_ets_free_allocation_t': 0.0}
        
        # TR-ETS covers all emissions for domestic production
        domestic_share = 1 - (self.params.eu_export_volume / self.params.plant_capacity_tpa)
        tr_ets_emissions = emissions['total_co2'] * domestic_share
        
        # Calculate free allocation
        free_allocation_total = self.calculate_tr_ets_free_allocation(year)
        free_allocation_domestic = free_allocation_total * domestic_share
        
        # Net emissions requiring allowance purchase
        net_emissions = max(0, tr_ets_emissions - free_allocation_domestic)
        
        # Get escalated TR-ETS price
        tr_ets_price_usd = self.get_escalated_price(
            self.params.tr_ets_price_usd_2026, 'tr_ets_price_usd', 2026, year
        )
        
        tr_ets_cost_usd = net_emissions * tr_ets_price_usd
        
        return {
            'tr_ets_cost_usd': tr_ets_cost_usd,
            'tr_ets_price_usd': tr_ets_price_usd,
            'tr_ets_emissions_t': tr_ets_emissions,
            'tr_ets_free_allocation_t': free_allocation_domestic,
            'tr_ets_net_emissions_t': net_emissions
        }
    
    def calculate_freight_phase_in_factor(self, year: int) -> float:
        """Calculate freight ETS phase-in factor (actual implementation)"""
        if year < 2024:
            return 0.0  # No freight ETS before 2024
        elif year >= 2026:
            return 1.0  # Full freight ETS from 2026 onwards
        else:
            # Use actual phase-in schedule
            return self.params.freight_ets_phase_in_schedule.get(year, 1.0)
    
    def calculate_freight_cost(self, year: int) -> Dict[str, float]:
        """Calculate freight ETS pass-through costs with actual phase-in schedule (USD)"""
        if year < 2024:
            return {'freight_cost_usd': 0.0, 'freight_phase_in_factor': 0.0}
        
        # Calculate phase-in factor
        phase_in_factor = self.calculate_freight_phase_in_factor(year)
        
        # Get escalated freight cost
        freight_cost_per_t_usd = self.get_escalated_price(
            self.params.freight_ets_pass_through_usd_2026, 'freight_ets_pass_through_usd', 2026, year
        )
        
        # Apply phase-in factor
        freight_cost_usd = self.params.eu_export_volume * freight_cost_per_t_usd * phase_in_factor
        
        return {
            'freight_cost_usd': freight_cost_usd,
            'freight_cost_per_t_usd': freight_cost_per_t_usd,
            'freight_phase_in_factor': phase_in_factor
        }
    
    def calculate_annual_costs(self, year: int) -> Dict[str, Any]:
        """Calculate all carbon costs for a given year"""
        emissions = self.calculate_co2_emissions(year)
        cbam_costs = self.calculate_cbam_cost(year, emissions)
        tr_ets_costs = self.calculate_tr_ets_cost(year, emissions)
        freight_costs = self.calculate_freight_cost(year)
        
        total_cost_usd = (
            cbam_costs['cbam_cost_usd'] + 
            tr_ets_costs['tr_ets_cost_usd'] + 
            freight_costs['freight_cost_usd']
        )
        
        cost_per_tonne_usd = total_cost_usd / self.params.plant_capacity_tpa
        
        return {
            'year': year,
            'emissions': emissions,
            'cbam': cbam_costs,
            'tr_ets': tr_ets_costs,
            'freight': freight_costs,
            'total_cost_usd': total_cost_usd,
            'cost_per_tonne_usd': cost_per_tonne_usd
        }
    
    def run_optimization(self) -> pd.DataFrame:
        """Run the full optimization for all years"""
        results = []
        
        for year in self.years:
            annual_result = self.calculate_annual_costs(year)
            results.append({
                'Year': year,
                'Total_CO2_t': annual_result['emissions']['total_co2'],
                'CBAM_Cost_USD': annual_result['cbam']['cbam_cost_usd'],
                'CBAM_Phase_In_Factor': annual_result['cbam'].get('cbam_phase_in_factor', 0.0),
                'TR_ETS_Cost_USD': annual_result['tr_ets']['tr_ets_cost_usd'],
                'TR_ETS_Free_Alloc_t': annual_result['tr_ets'].get('tr_ets_free_allocation_t', 0.0),
                'Freight_Cost_USD': annual_result['freight']['freight_cost_usd'],
                'Freight_Phase_In_Factor': annual_result['freight'].get('freight_phase_in_factor', 0.0),
                'Total_Cost_USD': annual_result['total_cost_usd'],
                'Cost_per_Tonne_USD': annual_result['cost_per_tonne_usd']
            })
            
            self.results[year] = annual_result
        
        return pd.DataFrame(results)
    
    def sensitivity_analysis(self, base_results: pd.DataFrame) -> pd.DataFrame:
        """Perform ±10% sensitivity analysis on key parameters"""
        sensitive_params = [
            'co2_process_intensity',
            'clinker_ratio',
            'alt_fuel_share',
            'renewable_el_share',
            'eu_export_volume',
            'cbam_price_usd_2026',
            'tr_ets_price_usd_2026',
            'eu_benchmark_t_co2_per_t_cement',
            'tr_ets_free_allocation_share'
        ]
        
        sensitivity_results = []
        base_npv = base_results['Total_Cost_USD'].sum()
        
        for param in sensitive_params:
            # +10% scenario
            high_params = CarbonParameters(**self.params.__dict__)
            setattr(high_params, param, getattr(high_params, param) * 1.1)
            high_optimizer = CarbonOptimizer(high_params)
            high_results = high_optimizer.run_optimization()
            high_npv = high_results['Total_Cost_USD'].sum()
            
            # -10% scenario
            low_params = CarbonParameters(**self.params.__dict__)
            setattr(low_params, param, getattr(low_params, param) * 0.9)
            low_optimizer = CarbonOptimizer(low_params)
            low_results = low_optimizer.run_optimization()
            low_npv = low_results['Total_Cost_USD'].sum()
            
            sensitivity_results.append({
                'Parameter': param,
                'Base_NPV_USD': base_npv,
                'High_NPV_USD': high_npv,
                'Low_NPV_USD': low_npv,
                'High_Impact_USD': high_npv - base_npv,
                'Low_Impact_USD': low_npv - base_npv,
                'Range_USD': high_npv - low_npv
            })
        
        return pd.DataFrame(sensitivity_results).sort_values('Range_USD', ascending=False)

def get_user_inputs() -> CarbonParameters:
    """Collect user inputs with fallback to synthetic data"""
    print("=== Batıçim Cement Carbon-Cost Optimizer ===")
    print("Please provide actual plant data (press Enter to use synthetic defaults):\n")
    
    # Plant capacity
    try:
        capacity_input = input("Plant cement capacity (t/yr) [1,600,000]: ")
        plant_capacity = float(capacity_input) if capacity_input else 1_600_000
    except (ValueError, EOFError):
        plant_capacity = 1_600_000
        print("Using synthetic data: 1,600,000 t/yr")
    
    # CO2 process intensity
    try:
        co2_input = input("CO2 process intensity (kg CO2/t clinker) [820]: ")
        co2_intensity = float(co2_input) if co2_input else 820
    except (ValueError, EOFError):
        co2_intensity = 820
        print("Using synthetic data: 820 kg CO2/t clinker")
    
    # Clinker ratio
    try:
        clinker_input = input("Clinker-to-cement ratio [0.78]: ")
        clinker_ratio = float(clinker_input) if clinker_input else 0.78
    except (ValueError, EOFError):
        clinker_ratio = 0.78
        print("Using synthetic data: 0.78")
    
    # EU export volume
    try:
        export_input = input("EU export volume (t/yr) [600,000]: ")
        eu_exports = float(export_input) if export_input else 600_000
    except (ValueError, EOFError):
        eu_exports = 600_000
        print("Using synthetic data: 600,000 t/yr")
    
    print("\nAll other parameters using synthetic defaults...")
    
    return CarbonParameters(
        plant_capacity_tpa=plant_capacity,
        co2_process_intensity=co2_intensity,
        clinker_ratio=clinker_ratio,
        eu_export_volume=eu_exports
    )

def main():
    """Main execution function"""
    # Use synthetic data for testing
    print("=== Batıçim Cement Carbon-Cost Optimizer ===")
    print("Running with synthetic data for testing...\n")
    
    params = CarbonParameters()
    
    # Run optimization
    print("=== Running Carbon Cost Optimization ===")
    optimizer = CarbonOptimizer(params)
    results = optimizer.run_optimization()
    
    # Display results
    print("\n=== 2024-2034 Carbon Cost Projections (USD) ===")
    print(results.round(2))
    
    # Summary statistics
    print(f"\n=== Summary ===")
    print(f"Total 11-year carbon cost: ${results['Total_Cost_USD'].sum():,.0f}")
    print(f"Average annual cost: ${results['Total_Cost_USD'].mean():,.0f}")
    print(f"Peak annual cost: ${results['Total_Cost_USD'].max():,.0f} (Year {results.loc[results['Total_Cost_USD'].idxmax(), 'Year']})")
    print(f"Average cost per tonne: ${results['Cost_per_Tonne_USD'].mean():.2f}")
    
    # Sensitivity analysis
    print("\n=== Sensitivity Analysis (±10% shocks) ===")
    sensitivity = optimizer.sensitivity_analysis(results)
    print(sensitivity[['Parameter', 'Range_USD']].round(0))
    
    # Save results
    results.to_csv('carbon_cost_projections.csv', index=False)
    sensitivity.to_csv('sensitivity_analysis.csv', index=False)
    
    print(f"\nResults saved to:")
    print(f"- carbon_cost_projections.csv")
    print(f"- sensitivity_analysis.csv")

if __name__ == "__main__":
    main()