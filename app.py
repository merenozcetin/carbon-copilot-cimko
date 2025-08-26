import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from carbon_optimizer import CarbonOptimizer, CarbonParameters
import json

# Sayfa ayarı
st.set_page_config(
    page_title="Azalt: Carbon Risk Co-Pilot - Çimko",  # 'Carbon Risk' orijinal bırakıldı
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Kurumsal stil
st.markdown("""
<style>
    .main > div { padding-top: 1rem; font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; }
    .main-header { background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%); color: white; padding: 2rem; border-radius: 10px; margin-bottom: 2rem; text-align: center; box-shadow: 0 4px 20px rgba(0,0,0,0.1); }
    .stMetric { background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%); border: none; padding: 1.5rem; border-radius: 12px; margin: 0.5rem 0; box-shadow: 0 2px 10px rgba(0,0,0,0.08); border-left: 4px solid #2a5298; }
    .css-1d391kg { background-color: #f8f9fa; border-right: 2px solid #e9ecef; }
    .chart-container { background: white; border-radius: 12px; padding: 1rem; margin: 1rem 0; box-shadow: 0 2px 15px rgba(0,0,0,0.05); border: 1px solid #e9ecef; }
    .subheader { color: #2c3e50; font-weight: 600; margin: 1.5rem 0 1rem 0; padding-bottom: 0.5rem; border-bottom: 2px solid #3498db; }
    .dataframe { border: none !important; border-radius: 8px; overflow: hidden; box-shadow: 0 2px 10px rgba(0,0,0,0.05); }
    .stButton > button { background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; border: none; border-radius: 8px; padding: 0.75rem 1.5rem; font-weight: 600; transition: all 0.3s ease; }
    .stButton > button:hover { transform: translateY(-2px); box-shadow: 0 4px 15px rgba(0,0,0,0.2); }
    .sidebar-section { background: white; border-radius: 8px; padding: 1rem; margin: 1rem 0; border-left: 4px solid #3498db; }
</style>
""", unsafe_allow_html=True)

def main():
    # Başlık
    st.markdown("""
    <div style="text-align: center; margin: 2rem 0; padding: 1rem 0;">
        <h1 style="color: #2c3e50; font-family: 'Segoe UI', sans-serif; font-weight: 300; font-size: 2.5rem; margin: 0; letter-spacing: 2px;">
            Azalt: Carbon Risk Co-Pilot - Çimko
        </h1>
        <p style="color: #7f8c8d; font-size: 1.1rem; margin: 0.5rem 0 0 0; font-weight: 400;">
            2024-2034 dönemi için kurumsal düzey karbon maliyeti modellemesi
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Kenar çubuğu – Parametreler
    st.sidebar.header("TESİS PARAMETRELERİ")
    
    # Üretim parametreleri
    with st.sidebar.expander("ÜRETİM PARAMETRELERİ", expanded=True):
        plant_capacity = st.number_input(
            "Tesis Kapasitesi (t çimento/yıl)", 
            min_value=500_000, 
            max_value=18_000_000, 
            value=1_759_584,
            step=50_000,
            help="Yıllık toplam çimento üretim kapasitesi"
        )
        
        clinker_ratio = st.slider(
            "Klinker/Çimento Oranı", 
            min_value=0.60, 
            max_value=0.90, 
            value=0.80,
            step=0.01,
            help="Çimentodaki klinker oranı (yüksek oran = daha fazla CO₂)"
        )
        
        eu_export_volume = st.number_input(
            "AB İhracat Miktarı (t/yıl)", 
            min_value=0, 
            max_value=int(plant_capacity), 
            value=min(650_000, int(plant_capacity)),
            step=25_000,
            help="AB'ye yıllık çimento ihracatı (CBAM kapsamında)"
        )
    
    # Emisyon parametreleri
    with st.sidebar.expander("EMİSYON PARAMETRELERİ", expanded=True):
        co2_process_intensity = st.slider(
            "CO₂ Proses Yoğunluğu (kg CO₂/t klinker)", 
            min_value=700, 
            max_value=950, 
            value=904,
            step=5,
            help="Kireçtaşı kalsinasyonundan kaynaklanan CO₂ emisyonu"
        )
        
        thermal_energy = st.slider(
            "Isıl Enerji (MJ/t klinker)", 
            min_value=3000, 
            max_value=4500, 
            value=3830,
            step=25,
            help="Ton başına klinker ısıl enerji tüketimi"
        )
        
        alt_fuel_share = st.slider(
            "Alternatif Yakıt Payı (%)", 
            min_value=0, 
            max_value=50, 
            value=0,
            step=1,
            help="Isıl enerjide alternatif yakıtların payı"
        ) / 100
        
        renewable_el_share = st.slider(
            "Yenilenebilir Elektrik Payı (%)", 
            min_value=0, 
            max_value=100, 
            value=30,
            step=1,
            help="Elektrik tüketiminde yenilenebilir kaynakların payı"
        ) / 100
        
        electricity_use = st.slider(
            "Elektrik Tüketimi (kWh/t çimento)", 
            min_value=50, 
            max_value=150, 
            value=78,
            step=2,
            help="Ton başına çimento elektrik tüketimi"
        )
    
    # Karbon fiyatlaması
    with st.sidebar.expander("KARBON FİYATLAMASI", expanded=True):
        cbam_price_2026 = st.slider(
            "CBAM Fiyatı 2026 (USD/t CO₂)", 
            min_value=50, 
            max_value=200, 
            value=90,
            step=5,
            help="2026 CBAM sertifika fiyatı"
        )
        
        tr_ets_price_2026 = st.slider(
            "TR-ETS Fiyatı 2026 (USD/t CO₂)", 
            min_value=5, 
            max_value=50, 
            value=9,
            step=1,
            help="2026 Türkiye ETS tahsis fiyatı"
        )
    
    # Ücretsiz tahsis
    with st.sidebar.expander("ÜCRETSİZ TAHSİS", expanded=True):
        eu_benchmark = st.slider(
            "AB Kıyas Değeri (t CO₂/t çimento)", 
            min_value=0.500, 
            max_value=0.900, 
            value=0.693,
            step=0.005,
            help="AB-ETS Faz 4 çimento kıyas değeri"
        )
        
        tr_ets_free_share = st.slider(
            "TR-ETS Ücretsiz Tahsis Oranı (%)", 
            min_value=0, 
            max_value=100, 
            value=100,
            step=1,
            help="TR-ETS ücretsiz tahsis payı"
        ) / 100
    
    # Parametre objesi
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
    
    # Optimizasyon
    optimizer = CarbonOptimizer(params)
    results = optimizer.run_optimization()
    
    # Göstergeler
    st.markdown('<div class="subheader">TEMEL PERFORMANS GÖSTERGELERİ</div>', unsafe_allow_html=True)
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            "Toplam 11 Yıllık Maliyet",
            f"${results['Total_Cost_USD'].sum():,.0f}",
            help="2024-2034 dönemi karbonla ilgili toplam maliyetler"
        )
    
    with col2:
        st.metric(
            "Ortalama Yıllık Maliyet",
            f"${results['Total_Cost_USD'].mean():,.0f}",
            help="Yıllık ortalama karbon maliyeti"
        )
    
    with col3:
        st.metric(
            "En Yüksek Yıllık Maliyet",
            f"${results['Total_Cost_USD'].max():,.0f}",
            help=f"En yüksek yıllık maliyet (Yıl {results.loc[results['Total_Cost_USD'].idxmax(), 'Year']})"
        )
    
    with col4:
        st.metric(
            "Ton Başına Maliyet",
            f"${results['Cost_per_Tonne_USD'].mean():.2f}",
            help="Üretilen çimento başına ortalama maliyet"
        )
    
    # Yıllık maliyet dağılımı grafiği
    st.markdown('<div class="subheader">YILLIK MALİYET DAĞILIMI</div>', unsafe_allow_html=True)
    
    fig_breakdown = go.Figure()
    fig_breakdown.add_trace(go.Scatter(x=results['Year'], y=results['CBAM_Cost_USD'], mode='lines+markers', name='CBAM Maliyeti', line=dict(color='#e74c3c', width=3)))
    fig_breakdown.add_trace(go.Scatter(x=results['Year'], y=results['TR_ETS_Cost_USD'], mode='lines+markers', name='TR-ETS Maliyeti', line=dict(color='#3498db', width=3)))
    fig_breakdown.add_trace(go.Scatter(x=results['Year'], y=results['Freight_Cost_USD'], mode='lines+markers', name='Navlun ETS Maliyeti', line=dict(color='#2ecc71', width=3)))
    fig_breakdown.add_trace(go.Scatter(x=results['Year'], y=results['Total_Cost_USD'], mode='lines+markers', name='Toplam Maliyet', line=dict(color='#2c3e50', width=4), yaxis='y2'))
    fig_breakdown.update_layout(
        title=dict(text="Karbon Maliyetlerinin Seyri 2024-2034", font=dict(size=20, family="Arial, sans-serif", color="#2c3e50"), x=0.5),
        xaxis_title="Yıl",
        yaxis_title="Bileşen Bazlı Maliyetler (USD)",
        yaxis2=dict(title="Toplam Maliyet (USD)", overlaying='y', side='right'),
        hovermode='x unified',
        height=500,
        showlegend=True,
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(family="Arial, sans-serif", color="#2c3e50"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5)
    )
    st.plotly_chart(fig_breakdown, use_container_width=True)
    
    # Ücretsiz tahsis etkisi
    st.markdown('<div class="subheader">ÜCRETSİZ TAHSİS ETKİSİ</div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    with col1:
        fig_cbam = go.Figure()
        fig_cbam.add_trace(go.Scatter(x=results['Year'], y=results['CBAM_Phase_In_Factor'], mode='lines+markers', name='CBAM Kademeli Geçiş Katsayısı', line=dict(color='#FF6B6B', width=3), fill='tozeroy'))
        fig_cbam.update_layout(title="CBAM Kademeli Geçiş Takvimi", xaxis_title="Yıl", yaxis_title="Kademeli Geçiş Katsayısı", yaxis=dict(range=[0, 1.1]), height=400)
        st.plotly_chart(fig_cbam, use_container_width=True)
    with col2:
        fig_tr_ets = go.Figure()
        fig_tr_ets.add_trace(go.Scatter(x=results['Year'], y=results['TR_ETS_Free_Alloc_t'], mode='lines+markers', name='TR-ETS Ücretsiz Tahsisler', line=dict(color='#4ECDC4', width=3), fill='tozeroy'))
        fig_tr_ets.update_layout(title="TR-ETS Ücretsiz Tahsisler (Yıllık)", xaxis_title="Yıl", yaxis_title="Ücretsiz Tahsis (t CO₂)", height=400)
        st.plotly_chart(fig_tr_ets, use_container_width=True)
    
    # Duyarlılık analizi
    st.markdown('<div class="subheader">DUYARLILIK ANALİZİ</div>', unsafe_allow_html=True)
    sensitivity = optimizer.sensitivity_analysis(results)
    fig_tornado = go.Figure()
    sensitivity_sorted = sensitivity.sort_values('Range_USD', key=abs, ascending=True)
    fig_tornado.add_trace(go.Bar(y=sensitivity_sorted['Parameter'], x=sensitivity_sorted['Low_Impact_USD'], orientation='h', name='Düşük Etki (-%10)', marker_color='#FF6B6B', text=[f"${x:,.0f}" for x in sensitivity_sorted['Low_Impact_USD']], textposition='inside'))
    fig_tornado.add_trace(go.Bar(y=sensitivity_sorted['Parameter'], x=sensitivity_sorted['High_Impact_USD'], orientation='h', name='Yüksek Etki (+%10)', marker_color='#4ECDC4', text=[f"${x:,.0f}" for x in sensitivity_sorted['High_Impact_USD']], textposition='inside'))
    fig_tornado.update_layout(title="Duyarlılık Tornado Grafiği (±%10 Parametre Değişimi)", xaxis_title="Toplam Maliyet Etkisi (USD)", yaxis_title="Parametre", barmode='relative', height=600, showlegend=True)
    st.plotly_chart(fig_tornado, use_container_width=True)
    
    # Ayrıntılı sonuçlar
    st.markdown('<div class="subheader">AYRINTILI SONUÇLAR</div>', unsafe_allow_html=True)
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
    
    # Dışa aktarma
    st.markdown('<div class="subheader">SONUÇLARI DIŞA AKTAR</div>', unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    with col1:
        csv = results.to_csv(index=False)
        st.download_button(label="Sonuçları CSV Olarak İndir", data=csv, file_name="carbon_cost_projections.csv", mime="text/csv")
    with col2:
        sensitivity_csv = sensitivity.to_csv(index=False)
        st.download_button(label="Duyarlılık Analizi CSV'sini İndir", data=sensitivity_csv, file_name="sensitivity_analysis.csv", mime="text/csv")
    
    # Varsayımlar
    st.markdown('<div class="subheader">TEMEL VARSAYIMLAR</div>', unsafe_allow_html=True)
    assumptions_text = """
    **KARBON FİYATLAMASI VARSAYIMLARI:**
    • CBAM sertifika fiyatları EU ETS eğilimini izler ve yıllık %5–7 artar  
    • TR-ETS fiyatları 2026'da 9 USD/tCO₂ ile başlar ve kademeli olarak AB seviyelerine yakınsar  
    • Navlun ETS 2024'te %40 ile başladı, 2025'te %70'e çıktı, 2026'dan itibaren tam uygulama
    
    **ÜRETİM VARSAYIMLARI:**
    • Tesis kapasite kullanım oranı mevcut seviyelerde sabit  
    • Klinker/çimento oranı mevcut ürün karmasını yansıtır  
    • AB ihracat hacmi mevcut pazar payını korur
    
    **EMİSYON VARSAYIMLARI:**
    • Proses emisyonları kireçtaşı kalsinasyonu stokiyometrisine dayanır  
    • Isıl enerji gereksinimleri mevcut fırın verimliliğini yansıtır  
    • Elektrik şebekesi karbon yoğunluğu ulusal enerji dönüşüm planlarını izler
    
    **MEVZUAT VARSAYIMLARI:**
    • CBAM uygulaması AB zaman çizelgesini (2026 tam uygulama) izler  
    • TR-ETS ücretsiz tahsisleri 2026'dan itibaren yılda %2,2 azalır  
    • AB kıyas değerleri Faz 4 seviyelerinde sabit kalır
    
    **FİNANSAL VARSAYIMLAR:**
    • USD döviz kuru EUR ve TRY karşısında görece sabit  
    • Modellenen sistemler dışında ek karbon vergileri/harçları yok  
    • Taşımacılık maliyetleri olası mod değişimlerini içermez
    """
    st.markdown(assumptions_text)
    
    # Alt bilgi
    st.markdown("---")
    st.markdown(
        "**AZALT: CARBON RISK CO-PILOT - ÇİMKO** | "
        "Kurumsal Karbon Analitiği Platformu | "
        "Ücretsiz tahsis mekanizmalarıyla kapsamlı CBAM, TR-ETS ve Navlun ETS modellemesi"
    )

if __name__ == "__main__":
    main()
