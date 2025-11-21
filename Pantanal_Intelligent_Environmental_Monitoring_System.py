# # 🔥 Sistema Inteligente de Monitoramento e Predição de Queimadas no Pantanal
# ## Aplicações em Aprendizado de Máquina - Ciência de Dados
# 
# ---
# 
# ### 📋 Sumário Executivo
# 
# **Contexto:** O Pantanal, maior planície alagável do mundo e patrimônio natural da humanidade, enfrentou em 2020 uma das piores temporadas de queimadas de sua história. Este projeto desenvolve um sistema inteligente de análise e predição utilizando dados geoespaciais reais de focos de calor.
# 
# **Objetivos:**
# - Analisar padrões espaço-temporais de queimadas no Pantanal em 2020
# - Identificar clusters naturais de focos com características similares
# - Desenvolver modelos preditivos para antecipação de ocorrências
# - Gerar insights acionáveis para políticas de prevenção e combate
# 
# **Metodologia:**
# - Análise Exploratória de Dados (EDA)
# - Aprendizado Não Supervisionado (K-Means, DBSCAN)
# - Aprendizado Supervisionado (Random Forest, XGBoost)
# - Visualização Geoespacial Avançada
# 
# ---

# ## 1️⃣ Configuração do Ambiente e Importação de Bibliotecas

# ### 1.1 Instalação de Dependências

# Instalar bibliotecas necessárias (executar apenas se não estiverem instaladas)
#!pip install -q geopandas folium plotly xgboost scikit-learn pandas numpy matplotlib seaborn

# ### 1.2 Importação de Bibliotecas

# Manipulação e análise de dados
# ------------------------------
# 1️⃣ Importação de Bibliotecas
# ------------------------------

# Manipulação e análise de dados
import pandas as pd
import numpy as np
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Banco de dados analítico em disco/memória (alta performance local)
import duckdb

# Visualização
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Análise geoespacial (sem GeoPandas para evitar GDAL)
# Se você tiver GeoPandas instalado, pode descomentar a linha abaixo.
# import geopandas as gpd

import folium
from folium.plugins import HeatMap, MarkerCluster

# Machine Learning - Pré-processamento
from sklearn.preprocessing import StandardScaler, LabelEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.impute import SimpleImputer

# Machine Learning - Algoritmos Não Supervisionados
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score

# Machine Learning - Algoritmos Supervisionados
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.tree import DecisionTreeClassifier
import xgboost as xgb

# Machine Learning - Métricas
from sklearn.metrics import (
    classification_report, confusion_matrix, accuracy_score,
    precision_score, recall_score, f1_score, roc_auc_score, roc_curve,
    mean_squared_error, mean_absolute_error, r2_score
)

# (opcional) apenas para printar versão
import sklearn

# (opcional) Spark + Sedona para Big Data
USE_SPARK = False
try:
    from pyspark.sql import SparkSession
    from sedona.register import SedonaRegistrator
    from sedona.utils import SedonaKryoRegistrator, KryoSerializer

    spark = (
        SparkSession.builder
        .appName("QueimadasPantanal2020_2024")
        .config("spark.serializer", KryoSerializer.getName)
        .config("spark.kryo.registrator", SedonaKryoRegistrator.getName)
        .getOrCreate()
    )
    SedonaRegistrator.registerAll(spark)
    USE_SPARK = True
    print("✅ Spark + Sedona inicializados (modo Big Data pronto).")
except ImportError:
    print("⚠️ Spark/Sedona não encontrados. Seguindo apenas com DuckDB + Pandas.")

# Configurações de visualização
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', 100)

print("✅ Todas as bibliotecas importadas com sucesso!")
print(f"📊 Versões principais:")
print(f"   - Pandas: {pd.__version__}")
print(f"   - NumPy: {np.__version__}")
print(f"   - Scikit-learn: {sklearn.__version__}")

# ---

# ## 2️⃣ Etapa 1: Carregamento e Exploração Inicial do Dataset
# ------------------------------
# 1️⃣ Importação de Bibliotecas
# ------------------------------

# Manipulação e análise de dados
import pandas as pd
import numpy as np
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Banco de dados analítico em disco/memória (alta performance local)
import duckdb

# Visualização
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Análise geoespacial (sem GeoPandas para evitar GDAL)
# Se você tiver GeoPandas instalado, pode descomentar a linha abaixo.
# import geopandas as gpd

import folium
from folium.plugins import HeatMap, MarkerCluster

# Machine Learning - Pré-processamento
from sklearn.preprocessing import StandardScaler, LabelEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.impute import SimpleImputer

# Machine Learning - Algoritmos Não Supervisionados
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score

# Machine Learning - Algoritmos Supervisionados
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.tree import DecisionTreeClassifier
import xgboost as xgb

# Machine Learning - Métricas
from sklearn.metrics import (
    classification_report, confusion_matrix, accuracy_score,
    precision_score, recall_score, f1_score, roc_auc_score, roc_curve,
    mean_squared_error, mean_absolute_error, r2_score
)

# (opcional) apenas para printar versão
import sklearn

# (opcional) Spark + Sedona para Big Data
USE_SPARK = False
try:
    from pyspark.sql import SparkSession
    from sedona.register import SedonaRegistrator
    from sedona.utils import SedonaKryoRegistrator, KryoSerializer

    spark = (
        SparkSession.builder
        .appName("QueimadasPantanal2020_2024")
        .config("spark.serializer", KryoSerializer.getName)
        .config("spark.kryo.registrator", SedonaKryoRegistrator.getName)
        .getOrCreate()
    )
    SedonaRegistrator.registerAll(spark)
    USE_SPARK = True
    print("✅ Spark + Sedona inicializados (modo Big Data pronto).")
except ImportError:
    print("⚠️ Spark/Sedona não encontrados. Seguindo apenas com DuckDB + Pandas.")

# Configurações de visualização
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', 100)

print("✅ Todas as bibliotecas importadas com sucesso!")
print(f"📊 Versões principais:")
print(f"   - Pandas: {pd.__version__}")
print(f"   - NumPy: {np.__version__}")
print(f"   - Scikit-learn: {sklearn.__version__}")

# Criar cópia para trabalho (preservar original)
df = df_original.copy()


# ------------------------------------------------------------------
# (Opcional) Carregar os mesmos dados via Spark/Sedona a partir do Parquet
# ------------------------------------------------------------------
if USE_SPARK:
    print("\n" + "=" * 80)
    print("🔥 CARREGANDO DADOS NO SPARK A PARTIR DO PARQUET")
    print("=" * 80)

    df_spark = spark.read.parquet(parquet_path)
    df_spark.createOrReplaceTempView("queimadas_spark")

    print(f"✅ DataFrame Spark carregado: {df_spark.count():,} linhas")
    print("📌 Schema Spark:")
    df_spark.printSchema()

    # Exemplo de consulta em Spark (Big Data)
    exemplo = spark.sql("""
        SELECT ano_dado, COUNT(*) AS focos
        FROM queimadas_spark
        GROUP BY ano_dado
        ORDER BY ano_dado
    """)
    exemplo.show()

    # Se você quiser jogar uma amostra para Pandas:
    # df_amostra = df_spark.sample(fraction=0.1, seed=42).toPandas()

# ### 2.2 Inspeção Inicial da Estrutura

print("=" * 80)
print("📋 INFORMAÇÕES ESTRUTURAIS DO DATASET")
print("=" * 80)
df.info()

# ### 2.3 Primeiras Linhas do Dataset

print("\n" + "=" * 80)
print("👀 PRIMEIRAS 10 LINHAS DO DATASET")
print("=" * 80)
display(df.head(10))

# ### 2.4 Estatísticas Descritivas

print("\n" + "=" * 80)
print("📊 ESTATÍSTICAS DESCRITIVAS - VARIÁVEIS NUMÉRICAS")
print("=" * 80)
display(df.describe().T)

# ### 2.5 Análise de Valores Ausentes

print("\n" + "=" * 80)
print("🔍 ANÁLISE DE VALORES AUSENTES")
print("=" * 80)

missing_data = pd.DataFrame({
    'Coluna': df.columns,
    'Valores_Ausentes': df.isnull().sum(),
    'Percentual (%)': (df.isnull().sum() / len(df)) * 100
})
missing_data = missing_data[missing_data['Valores_Ausentes'] > 0].sort_values('Valores_Ausentes', ascending=False)

if len(missing_data) > 0:
    display(missing_data)
    
    # Visualizar valores ausentes
    plt.figure(figsize=(12, 6))
    sns.barplot(data=missing_data, x='Coluna', y='Percentual (%)', palette='Reds_r')
    plt.title('Percentual de Valores Ausentes por Variável', fontsize=14, fontweight='bold')
    plt.xlabel('Variáveis')
    plt.ylabel('Percentual de Dados Ausentes (%)')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.show()
else:
    print("✅ Nenhum valor ausente detectado no dataset!")

# ### 2.6 Análise de Duplicatas

duplicatas = df.duplicated().sum()
print(f"\n🔍 Registros duplicados encontrados: {duplicatas:,}")

if duplicatas > 0:
    print(f"⚠️  Percentual de duplicatas: {(duplicatas/len(df)*100):.2f}%")
    print("   → Será necessário tratamento na etapa de pré-processamento")
else:
    print("✅ Nenhuma duplicata encontrada!")

# ### 2.7 Identificação de Tipos de Variáveis

print("\n" + "=" * 80)
print("🏷️  CLASSIFICAÇÃO DAS VARIÁVEIS")
print("=" * 80)

# Identificar colunas por tipo
colunas_numericas = df.select_dtypes(include=[np.number]).columns.tolist()
colunas_categoricas = df.select_dtypes(include=['object']).columns.tolist()
colunas_datetime = [col for col in df.columns if 'data' in col.lower() or 'date' in col.lower()]

print(f"\n📊 Variáveis Numéricas ({len(colunas_numericas)}):")
for col in colunas_numericas:
    print(f"   • {col}")

print(f"\n🏷️  Variáveis Categóricas ({len(colunas_categoricas)}):")
for col in colunas_categoricas:
    print(f"   • {col} - {df[col].nunique()} valores únicos")

print(f"\n📅 Possíveis Variáveis Temporais:")
for col in colunas_datetime:
    print(f"   • {col}")

# ---

# ## 3️⃣ Etapa 2: Análise Exploratória de Dados Aprofundada

print("\n" + "=" * 80)
print("🔬 INICIANDO ANÁLISE EXPLORATÓRIA APROFUNDADA")
print("=" * 80)

# ### 3.1 Análise Temporal

# Identificar e converter colunas de data
if 'data_hora_gmt' in df.columns or 'datahora' in df.columns:
    col_data = 'data_hora_gmt' if 'data_hora_gmt' in df.columns else 'datahora'
    df[col_data] = pd.to_datetime(df[col_data], errors='coerce')
    
    # Extrair componentes temporais
    df['data'] = df[col_data].dt.date
    df['ano'] = df[col_data].dt.year
    df['mes'] = df[col_data].dt.month
    df['dia'] = df[col_data].dt.day
    df['dia_semana'] = df[col_data].dt.dayofweek
    df['nome_dia_semana'] = df[col_data].dt.day_name()
    df['nome_mes'] = df[col_data].dt.month_name()
    df['dia_do_ano'] = df[col_data].dt.dayofyear
    df['semana_do_ano'] = df[col_data].dt.isocalendar().week
    
    # Definir estações do ano (Hemisfério Sul)
    def definir_estacao(mes):
        if mes in [12, 1, 2]:
            return 'Verão'
        elif mes in [3, 4, 5]:
            return 'Outono'
        elif mes in [6, 7, 8]:
            return 'Inverno'
        else:
            return 'Primavera'
    
    df['estacao'] = df['mes'].apply(definir_estacao)
    
    print(f"✅ Análise temporal configurada para: {col_data}")
    print(f"   Período: {df['data'].min()} até {df['data'].max()}")

# ### 3.2 Séries Temporais de Ocorrências

# Ocorrências diárias
ocorrencias_diarias = df.groupby('data').size().reset_index(name='ocorrencias')
ocorrencias_diarias['data'] = pd.to_datetime(ocorrencias_diarias['data'])

# Visualização de série temporal
fig = make_subplots(
    rows=2, cols=1,
    subplot_titles=('Série Temporal de Focos de Queimadas - 2020 (Diário)',
                   'Tendência Semanal (Média Móvel 7 dias)'),
    vertical_spacing=0.12
)

# Gráfico diário
fig.add_trace(
    go.Scatter(x=ocorrencias_diarias['data'], y=ocorrencias_diarias['ocorrencias'],
               mode='lines', name='Ocorrências Diárias',
               line=dict(color='orangered', width=1)),
    row=1, col=1
)

# Média móvel de 7 dias
ocorrencias_diarias['media_movel_7d'] = ocorrencias_diarias['ocorrencias'].rolling(window=7).mean()
fig.add_trace(
    go.Scatter(x=ocorrencias_diarias['data'], y=ocorrencias_diarias['media_movel_7d'],
               mode='lines', name='Média Móvel (7 dias)',
               line=dict(color='darkred', width=3)),
    row=2, col=1
)

fig.update_xaxes(title_text="Data", row=2, col=1)
fig.update_yaxes(title_text="Número de Focos", row=1, col=1)
fig.update_yaxes(title_text="Número de Focos", row=2, col=1)

fig.update_layout(height=700, title_text="📈 Análise Temporal de Queimadas no Pantanal - 2020",
                  showlegend=True, hovermode='x unified')
fig.show()

# Estatísticas temporais
print("\n📊 ESTATÍSTICAS TEMPORAIS:")
print(f"   • Total de focos detectados: {len(df):,}")
print(f"   • Média diária: {ocorrencias_diarias['ocorrencias'].mean():.1f} focos/dia")
print(f"   • Mediana diária: {ocorrencias_diarias['ocorrencias'].median():.1f} focos/dia")
print(f"   • Dia com mais focos: {ocorrencias_diarias.loc[ocorrencias_diarias['ocorrencias'].idxmax(), 'data']} ({ocorrencias_diarias['ocorrencias'].max():,} focos)")
print(f"   • Dia com menos focos: {ocorrencias_diarias.loc[ocorrencias_diarias['ocorrencias'].idxmin(), 'data']} ({ocorrencias_diarias['ocorrencias'].min():,} focos)")

# ### 3.3 Análise por Mês e Estação

# Agregar por mês
ocorrencias_mensais = df.groupby(['mes', 'nome_mes']).size().reset_index(name='ocorrencias')
ocorrencias_mensais = ocorrencias_mensais.sort_values('mes')

# Agregar por estação
ocorrencias_estacao = df.groupby('estacao').size().reset_index(name='ocorrencias')

# Visualização
fig = make_subplots(
    rows=1, cols=2,
    subplot_titles=('Ocorrências por Mês', 'Ocorrências por Estação do Ano'),
    specs=[[{'type': 'bar'}, {'type': 'pie'}]]
)

# Gráfico mensal
fig.add_trace(
    go.Bar(x=ocorrencias_mensais['nome_mes'], y=ocorrencias_mensais['ocorrencias'],
           marker_color='orangered', name='Mensal'),
    row=1, col=1
)

# Gráfico por estação
fig.add_trace(
    go.Pie(labels=ocorrencias_estacao['estacao'], values=ocorrencias_estacao['ocorrencias'],
           hole=0.4, marker_colors=['#FFD700', '#FF8C00', '#8B4513', '#90EE90']),
    row=1, col=2
)

fig.update_xaxes(title_text="Mês", row=1, col=1, tickangle=-45)
fig.update_yaxes(title_text="Número de Focos", row=1, col=1)

fig.update_layout(height=500, title_text="📅 Distribuição Temporal das Queimadas",
                  showlegend=False)
fig.show()

# Ranking de meses
print("\n🔥 RANKING DE MESES MAIS CRÍTICOS:")
top_meses = ocorrencias_mensais.sort_values('ocorrencias', ascending=False)
for idx, row in top_meses.iterrows():
    percentual = (row['ocorrencias'] / len(df)) * 100
    print(f"   {row['mes']:2d}. {row['nome_mes']:10s} - {row['ocorrencias']:7,} focos ({percentual:5.2f}%)")

# ### 3.4 Análise Espacial - Coordenadas Geográficas

# Verificar presença de coordenadas
if 'latitude' in df.columns and 'longitude' in df.columns:
    
    print("\n🗺️  ANÁLISE GEOESPACIAL:")
    print(f"   • Latitude - Mín: {df['latitude'].min():.4f}, Máx: {df['latitude'].max():.4f}")
    print(f"   • Longitude - Mín: {df['longitude'].min():.4f}, Máx: {df['longitude'].max():.4f}")
    
    # Remover coordenadas inválidas
    df_geo = df[(df['latitude'].notna()) & (df['longitude'].notna())]
    df_geo = df_geo[(df_geo['latitude'] >= -90) & (df_geo['latitude'] <= 90)]
    df_geo = df_geo[(df_geo['longitude'] >= -180) & (df_geo['longitude'] <= 180)]
    
    print(f"   • Registros válidos para análise espacial: {len(df_geo):,} ({len(df_geo)/len(df)*100:.2f}%)")
    
    # Scatter plot das coordenadas
    fig = px.scatter(df_geo.sample(min(50000, len(df_geo))),  # Amostra para performance
                     x='longitude', y='latitude',
                     color='mes', size_max=5,
                     title='🗺️ Distribuição Espacial dos Focos de Queimadas no Pantanal',
                     labels={'longitude': 'Longitude', 'latitude': 'Latitude', 'mes': 'Mês'},
                     color_continuous_scale='YlOrRd',
                     height=600)
    
    fig.update_traces(marker=dict(size=3, opacity=0.6))
    fig.update_layout(xaxis_title="Longitude", yaxis_title="Latitude")
    fig.show()

# ### 3.5 Análise por Estado e Município

if 'estado' in df.columns:
    # Análise por estado
    ocorrencias_estado = df.groupby('estado').size().reset_index(name='ocorrencias')
    ocorrencias_estado = ocorrencias_estado.sort_values('ocorrencias', ascending=False)
    
    print("\n🏛️  RANKING DE ESTADOS MAIS AFETADOS:")
    for idx, row in ocorrencias_estado.head(10).iterrows():
        percentual = (row['ocorrencias'] / len(df)) * 100
        print(f"   • {row['estado']:20s} - {row['ocorrencias']:7,} focos ({percentual:5.2f}%)")
    
    # Visualização
    fig = px.bar(ocorrencias_estado.head(15), x='estado', y='ocorrencias',
                 title='🏛️ Estados com Maior Número de Focos de Queimadas',
                 labels={'estado': 'Estado', 'ocorrencias': 'Número de Focos'},
                 color='ocorrencias', color_continuous_scale='Reds',
                 height=500)
    fig.update_layout(xaxis_tickangle=-45)
    fig.show()

if 'municipio' in df.columns:
    # Top municípios
    ocorrencias_municipio = df.groupby('municipio').size().reset_index(name='ocorrencias')
    ocorrencias_municipio = ocorrencias_municipio.sort_values('ocorrencias', ascending=False)
    
    print("\n🏘️  TOP 15 MUNICÍPIOS MAIS AFETADOS:")
    for idx, row in ocorrencias_municipio.head(15).iterrows():
        percentual = (row['ocorrencias'] / len(df)) * 100
        print(f"   • {row['municipio']:30s} - {row['ocorrencias']:6,} focos ({percentual:5.2f}%)")

# ### 3.6 Análise de Intensidade (FRP - Fire Radiative Power)

if 'frp' in df.columns or 'potencia_fogo' in df.columns:
    col_frp = 'frp' if 'frp' in df.columns else 'potencia_fogo'
    
    print(f"\n🔥 ANÁLISE DE INTENSIDADE DO FOGO ({col_frp}):")
    print(f"   • Média: {df[col_frp].mean():.2f} MW")
    print(f"   • Mediana: {df[col_frp].median():.2f} MW")
    print(f"   • Desvio Padrão: {df[col_frp].std():.2f} MW")
    print(f"   • Mínimo: {df[col_frp].min():.2f} MW")
    print(f"   • Máximo: {df[col_frp].max():.2f} MW")
    
    # Distribuição do FRP
    fig = make_subplots(rows=1, cols=2,
                        subplot_titles=('Distribuição da Potência Radiativa do Fogo',
                                       'Boxplot - Detecção de Outliers'))
    
    fig.add_trace(go.Histogram(x=df[col_frp], nbinsx=50, name='FRP',
                               marker_color='orangered'), row=1, col=1)
    
    fig.add_trace(go.Box(y=df[col_frp], name='FRP',
                         marker_color='orangered', boxmean='sd'), row=1, col=2)
    
    fig.update_xaxes(title_text="Potência Radiativa (MW)", row=1, col=1)
    fig.update_yaxes(title_text="Frequência", row=1, col=1)
    fig.update_yaxes(title_text="Potência Radiativa (MW)", row=1, col=2)
    
    fig.update_layout(height=400, showlegend=False,
                      title_text="📊 Análise da Intensidade dos Focos de Queimadas")
    fig.show()

# ### 3.7 Matriz de Correlação

# Selecionar apenas colunas numéricas relevantes
colunas_correlacao = [col for col in colunas_numericas if col in df.columns]
if len(colunas_correlacao) > 1:
    correlacao = df[colunas_correlacao].corr()
    
    plt.figure(figsize=(12, 10))
    mask = np.triu(np.ones_like(correlacao, dtype=bool))
    sns.heatmap(correlacao, mask=mask, annot=True, fmt='.2f', cmap='RdYlBu_r',
                center=0, square=True, linewidths=1, cbar_kws={"shrink": 0.8})
    plt.title('🔗 Matriz de Correlação entre Variáveis Numéricas', 
              fontsize=14, fontweight='bold', pad=20)
    plt.tight_layout()
    plt.show()

print("\n✅ Análise Exploratória de Dados concluída!")

# ---

# ## 4️⃣ Etapa 3: Pré-processamento e Feature Engineering

print("\n" + "=" * 80)
print("⚙️  INICIANDO PRÉ-PROCESSAMENTO E ENGENHARIA DE FEATURES")
print("=" * 80)

# ### 4.1 Tratamento de Valores Ausentes

# Criar cópia para pré-processamento
df_processed = df.copy()

# Imputação de valores ausentes em variáveis numéricas
if df_processed[colunas_numericas].isnull().sum().sum() > 0:
    print("\n🔧 Tratando valores ausentes em variáveis numéricas...")
    imputer_num = SimpleImputer(strategy='median')
    df_processed[colunas_numericas] = imputer_num.fit_transform(df_processed[colunas_numericas])
    print("   ✅ Imputação com mediana aplicada")

# Imputação de valores ausentes em variáveis categóricas
cols_cat_com_missing = [col for col in colunas_categoricas if df_processed[col].isnull().sum() > 0]
if len(cols_cat_com_missing) > 0:
    print("\n🔧 Tratando valores ausentes em variáveis categóricas...")
    for col in cols_cat_com_missing:
        df_processed[col].fillna('DESCONHECIDO', inplace=True)
    print(f"   ✅ {len(cols_cat_com_missing)} colunas categóricas tratadas")

# ### 4.2 Tratamento de Outliers

# Identificar outliers usando IQR
def identificar_outliers_iqr(serie):
    Q1 = serie.quantile(0.25)
    Q3 = serie.quantile(0.75)
    IQR = Q3 - Q1
    limite_inferior = Q1 - 1.5 * IQR
    limite_superior = Q3 + 1.5 * IQR
    return (serie < limite_inferior) | (serie > limite_superior)

# Análise de outliers nas principais variáveis
print("\n🔍 ANÁLISE DE OUTLIERS:")
if col_frp in df_processed.columns:
    outliers_frp = identificar_outliers_iqr(df_processed[col_frp])
    print(f"   • {col_frp}: {outliers_frp.sum():,} outliers ({outliers_frp.sum()/len(df_processed)*100:.2f}%)")
    
    # Para queimadas, valores extremos podem ser reais (incêndios severos)
    # Vamos manter mas criar flag
    df_processed['is_outlier_frp'] = outliers_frp.astype(int)

# ### 4.3 Feature Engineering - Criação de Novas Variáveis

print("\n🛠️  ENGENHARIA DE FEATURES:")

# 1. Features temporais já criadas: dia_semana, mes, estacao, etc.

# 2. Flag de período crítico (meses de seca - julho a outubro)
df_processed['periodo_critico'] = df_processed['mes'].apply(
    lambda x: 1 if x in [7, 8, 9, 10] else 0
)
print("   ✅ Feature 'periodo_critico' criada (meses de seca)")

# 3. Densidade temporal (focos por dia)
densidade_temporal = df_processed.groupby('data').size().to_dict()
df_processed['densidade_diaria'] = df_processed['data'].map(densidade_temporal)
print("   ✅ Feature 'densidade_diaria' criada")

# 4. Flag de fim de semana
df_processed['fim_de_semana'] = df_processed['dia_semana'].apply(
    lambda x: 1 if x >= 5 else 0
)
print("   ✅ Feature 'fim_de_semana' criada")

# 5. Classificação de intensidade do fogo
if col_frp in df_processed.columns:
    def classificar_intensidade(frp):
        if frp < 10:
            return 'Baixa'
        elif frp < 50:
            return 'Média'
        elif frp < 100:
            return 'Alta'
        else:
            return 'Muito Alta'
    
    df_processed['intensidade_classe'] = df_processed[col_frp].apply(classificar_intensidade)
    print("   ✅ Feature 'intensidade_classe' criada")

# 6. Coordenadas arredondadas para análise de hotspots
if 'latitude' in df_processed.columns and 'longitude' in df_processed.columns:
    df_processed['lat_round'] = df_processed['latitude'].round(2)
    df_processed['lon_round'] = df_processed['longitude'].round(2)
    
    # Criar identificador de grid
    df_processed['grid_id'] = (df_processed['lat_round'].astype(str) + '_' + 
                                df_processed['lon_round'].astype(str))
    
    # Densidade espacial
    densidade_espacial = df_processed.groupby('grid_id').size().to_dict()
    df_processed['densidade_espacial'] = df_processed['grid_id'].map(densidade_espacial)
    print("   ✅ Features geoespaciais criadas (grid, densidade espacial)")

print(f"\n📊 Dataset após feature engineering: {df_processed.shape}")

# ### 4.4 Codificação de Variáveis Categóricas

# Label Encoding para variáveis categóricas ordinais
label_encoders = {}

if 'intensidade_classe' in df_processed.columns:
    le_intensidade = LabelEncoder()
    df_processed['intensidade_classe_encoded'] = le_intensidade.fit_transform(
        df_processed['intensidade_classe']
    )
    label_encoders['intensidade_classe'] = le_intensidade
    print("\n✅ Label Encoding aplicado em 'intensidade_classe'")

# Para estado e município, usaremos frequência (já que há muitas categorias)
if 'estado' in df_processed.columns:
    estado_freq = df_processed['estado'].value_counts(normalize=True).to_dict()
    df_processed['estado_freq'] = df_processed['estado'].map(estado_freq)
    print("✅ Frequency encoding aplicado em 'estado'")

if 'municipio' in df_processed.columns:
    municipio_freq = df_processed['municipio'].value_counts(normalize=True).to_dict()
    df_processed['municipio_freq'] = df_processed['municipio'].map(municipio_freq)
    print("✅ Frequency encoding aplicado em 'municipio'")

# ### 4.5 Normalização de Features

# Selecionar features numéricas para normalização
features_para_normalizar = ['latitude', 'longitude', 'densidade_diaria', 
                            'densidade_espacial']
features_para_normalizar = [f for f in features_para_normalizar if f in df_processed.columns]

if col_frp in df_processed.columns:
    features_para_normalizar.append(col_frp)

# Aplicar MinMaxScaler
scaler = MinMaxScaler()
df_processed[features_para_normalizar] = scaler.fit_transform(
    df_processed[features_para_normalizar]
)

print(f"\n✅ Normalização aplicada em {len(features_para_normalizar)} features")
print(f"   Features normalizadas: {', '.join(features_para_normalizar)}")

print("\n✅ Pré-processamento concluído!")

# ---

# ## 5️⃣ Etapa 4: Análise de Clusterização (Aprendizado Não Supervisionado)

print("\n" + "=" * 80)
print("🔬 INICIANDO ANÁLISE DE CLUSTERIZAÇÃO")
print("=" * 80)

# ### 5.1 Preparação dos Dados para Clustering

# Selecionar features para clustering
features_clustering = ['latitude', 'longitude', 'mes', 'dia_do_ano']

if col_frp in df_processed.columns:
    features_clustering.append(col_frp)
if 'densidade_espacial' in df_processed.columns:
    features_clustering.append('densidade_espacial')
if 'densidade_diaria' in df_processed.columns:
    features_clustering.append('densidade_diaria')

# Criar dataset para clustering (remover NaNs)
df_clustering = df_processed[features_clustering].dropna()

# Padronizar para clustering
scaler_clustering = StandardScaler()
X_clustering = scaler_clustering.fit_transform(df_clustering)

print(f"📊 Dataset para clustering:")
print(f"   • Amostras: {X_clustering.shape[0]:,}")
print(f"   • Features: {X_clustering.shape[1]}")
print(f"   • Features utilizadas: {', '.join(features_clustering)}")

# ### 5.2 Método do Cotovelo (Elbow Method)

print("\n🔍 Determinando número ótimo de clusters...")

# Testar diferentes números de clusters
K_range = range(2, 11)
inertias = []
silhouette_scores = []

for k in K_range:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    
    # Usar amostra para acelerar processamento
    sample_size = min(50000, len(X_clustering))
    sample_indices = np.random.choice(len(X_clustering), sample_size, replace=False)
    X_sample = X_clustering[sample_indices]
    
    kmeans.fit(X_sample)
    inertias.append(kmeans.inertia_)
    silhouette_scores.append(silhouette_score(X_sample, kmeans.labels_))
    
    print(f"   K={k}: Inércia={kmeans.inertia_:.2f}, Silhueta={silhouette_scores[-1]:.3f}")

# Visualizar método do cotovelo e silhueta
fig = make_subplots(rows=1, cols=2,
                    subplot_titles=('Método do Cotovelo', 'Coeficiente de Silhueta'))

fig.add_trace(go.Scatter(x=list(K_range), y=inertias, mode='lines+markers',
                         line=dict(color='orangered', width=3),
                         marker=dict(size=10)), row=1, col=1)

fig.add_trace(go.Scatter(x=list(K_range), y=silhouette_scores, mode='lines+markers',
                         line=dict(color='green', width=3),
                         marker=dict(size=10)), row=1, col=2)

fig.update_xaxes(title_text="Número de Clusters (K)", row=1, col=1)
fig.update_xaxes(title_text="Número de Clusters (K)", row=1, col=2)
fig.update_yaxes(title_text="Inércia (WCSS)", row=1, col=1)
fig.update_yaxes(title_text="Coeficiente de Silhueta", row=1, col=2)

fig.update_layout(height=400, title_text="📊 Determinação do Número Ótimo de Clusters",
                  showlegend=False)
fig.show()

# ### 5.3 Aplicação do K-Means com K Ótimo

# Determinar K ótimo (maior silhueta)
k_otimo = K_range[np.argmax(silhouette_scores)]
print(f"\n🎯 Número ótimo de clusters selecionado: K = {k_otimo}")
print(f"   (baseado no maior coeficiente de silhueta: {max(silhouette_scores):.3f})")

# Treinar modelo final com todos os dados
kmeans_final = KMeans(n_clusters=k_otimo, random_state=42, n_init=20)

# Usar amostra estratificada para treino
sample_size_final = min(100000, len(X_clustering))
sample_indices_final = np.random.choice(len(X_clustering), sample_size_final, replace=False)
X_sample_final = X_clustering[sample_indices_final]

print(f"\n⏳ Treinando K-Means com {sample_size_final:,} amostras...")
kmeans_final.fit(X_sample_final)

# Predizer clusters para todas as amostras
print("⏳ Atribuindo clusters a todas as observações...")
clusters = kmeans_final.predict(X_clustering)
df_clustering['cluster'] = clusters

# Métricas de qualidade
silhouette_avg = silhouette_score(X_sample_final, kmeans_final.predict(X_sample_final))
davies_bouldin = davies_bouldin_score(X_sample_final, kmeans_final.predict(X_sample_final))
calinski_harabasz = calinski_harabasz_score(X_sample_final, kmeans_final.predict(X_sample_final))

print(f"\n📊 MÉTRICAS DE QUALIDADE DO CLUSTERING:")
print(f"   • Coeficiente de Silhueta: {silhouette_avg:.3f} (quanto maior, melhor)")
print(f"   • Índice Davies-Bouldin: {davies_bouldin:.3f} (quanto menor, melhor)")
print(f"   • Índice Calinski-Harabasz: {calinski_harabasz:.2f} (quanto maior, melhor)")

# Adicionar clusters ao dataframe processado
df_processed.loc[df_clustering.index, 'cluster'] = clusters

# ### 5.4 Análise e Interpretação dos Clusters

print("\n" + "=" * 80)
print("🔍 ANÁLISE DETALHADA DOS CLUSTERS")
print("=" * 80)

# Estatísticas por cluster
for cluster_id in range(k_otimo):
    cluster_data = df_clustering[df_clustering['cluster'] == cluster_id]
    n_amostras = len(cluster_data)
    percentual = (n_amostras / len(df_clustering)) * 100
    
    print(f"\n{'='*60}")
    print(f"📍 CLUSTER {cluster_id} - {n_amostras:,} focos ({percentual:.2f}%)")
    print(f"{'='*60}")
    
    # Estatísticas das features originais
    for feat in features_clustering:
        if feat in df_processed.columns:
            valores = df_processed.loc[cluster_data.index, feat]
            print(f"   • {feat:25s}: μ={valores.mean():8.2f}, σ={valores.std():8.2f}, "
                  f"min={valores.min():8.2f}, max={valores.max():8.2f}")
    
    # Distribuição temporal
    if 'mes' in cluster_data.columns:
        mes_predominante = df_processed.loc[cluster_data.index, 'nome_mes'].mode()[0]
        print(f"\n   🗓️  Mês predominante: {mes_predominante}")
    
    # Distribuição espacial
    if 'latitude' in cluster_data.columns and 'longitude' in cluster_data.columns:
        lat_centro = df_processed.loc[cluster_data.index, 'latitude'].mean()
        lon_centro = df_processed.loc[cluster_data.index, 'longitude'].mean()
        print(f"   🗺️  Centro geográfico: ({lat_centro:.2f}, {lon_centro:.2f})")
    
    # Intensidade média
    if col_frp in df_processed.columns:
        frp_medio = df_processed.loc[cluster_data.index, col_frp].mean()
        print(f"   🔥 Intensidade média (FRP): {frp_medio:.2f} MW")

# ### 5.5 Visualização dos Clusters

# Visualização geoespacial dos clusters
if 'latitude' in df_clustering.columns and 'longitude' in df_clustering.columns:
    
    # Amostrar para visualização
    df_viz_sample = df_clustering.sample(min(20000, len(df_clustering)))
    
    # Recuperar coordenadas originais
    lat_original = df_processed.loc[df_viz_sample.index, 'latitude']
    lon_original = df_processed.loc[df_viz_sample.index, 'longitude']
    
    fig = px.scatter_mapbox(
        df_viz_sample,
        lat=lat_original,
        lon=lon_original,
        color='cluster',
        color_continuous_scale='Viridis',
        zoom=4,
        height=600,
        title='🗺️ Distribuição Espacial dos Clusters de Queimadas',
        labels={'cluster': 'Cluster'}
    )
    
    fig.update_layout(mapbox_style="open-street-map")
    fig.update_traces(marker=dict(size=5, opacity=0.6))
    fig.show()

# Distribuição dos clusters ao longo do tempo
cluster_temporal = df_processed[df_processed['cluster'].notna()].groupby(
    ['mes', 'cluster']
).size().reset_index(name='ocorrencias')

fig = px.bar(cluster_temporal, x='mes', y='ocorrencias', color='cluster',
             title='📊 Distribuição Temporal dos Clusters',
             labels={'mes': 'Mês', 'ocorrencias': 'Número de Focos', 'cluster': 'Cluster'},
             barmode='stack', height=500)
fig.show()

# ### 5.6 DBSCAN - Clustering Baseado em Densidade

print("\n" + "=" * 80)
print("🔬 APLICANDO DBSCAN (Density-Based Clustering)")
print("=" * 80)

# Usar apenas features espaciais para DBSCAN
features_dbscan = ['latitude', 'longitude']
X_dbscan = df_clustering[features_dbscan].values

# Usar amostra para DBSCAN (é computacionalmente intensivo)
sample_size_dbscan = min(30000, len(X_dbscan))
sample_indices_dbscan = np.random.choice(len(X_dbscan), sample_size_dbscan, replace=False)
X_dbscan_sample = X_dbscan[sample_indices_dbscan]

print(f"⏳ Executando DBSCAN em {sample_size_dbscan:,} amostras...")

# Parâmetros DBSCAN (ajustar eps baseado na escala dos dados)
dbscan = DBSCAN(eps=0.3, min_samples=50)
clusters_dbscan = dbscan.fit_predict(X_dbscan_sample)

# Análise dos resultados
n_clusters_dbscan = len(set(clusters_dbscan)) - (1 if -1 in clusters_dbscan else 0)
n_noise = list(clusters_dbscan).count(-1)

print(f"\n📊 RESULTADOS DO DBSCAN:")
print(f"   • Número de clusters encontrados: {n_clusters_dbscan}")
print(f"   • Pontos de ruído (outliers): {n_noise:,} ({n_noise/len(clusters_dbscan)*100:.2f}%)")
print(f"   • Pontos em clusters: {sample_size_dbscan - n_noise:,}")

# Visualizar clusters DBSCAN
if n_clusters_dbscan > 0:
    df_dbscan_viz = pd.DataFrame({
        'latitude': X_dbscan_sample[:, 0],
        'longitude': X_dbscan_sample[:, 1],
        'cluster': clusters_dbscan
    })
    
    fig = px.scatter(df_dbscan_viz, x='longitude', y='latitude', color='cluster',
                     title='🗺️ DBSCAN: Clusters de Densidade Espacial',
                     labels={'cluster': 'Cluster (-1 = Ruído)'},
                     color_continuous_scale='Viridis',
                     height=600)
    fig.update_traces(marker=dict(size=3, opacity=0.6))
    fig.show()

print("\n✅ Análise de Clusterização concluída!")

# ---

# ## 6️⃣ Etapa 5: Modelagem Preditiva (Aprendizado Supervisionado)

print("\n" + "=" * 80)
print("🤖 INICIANDO MODELAGEM PREDITIVA")
print("=" * 80)

# ### 6.1 Definição do Problema e Preparação dos Dados

# Vamos criar um problema de classificação: predizer se um foco terá alta intensidade
# Definir limiar para classificação (baseado no terceiro quartil do FRP)

if col_frp in df_processed.columns:
    
    # Remover outliers extremos para melhor modelagem
    df_ml = df_processed[df_processed[col_frp] <= df_processed[col_frp].quantile(0.95)].copy()
    
    # Definir target: alta intensidade
    limiar_intensidade = df_ml[col_frp].quantile(0.75)
    df_ml['alta_intensidade'] = (df_ml[col_frp] > limiar_intensidade).astype(int)
    
    print(f"🎯 PROBLEMA DE CLASSIFICAÇÃO: Predizer Alta Intensidade de Queimadas")
    print(f"   • Limiar definido: {limiar_intensidade:.2f} MW (Q3)")
    print(f"   • Classe 0 (Baixa/Média): {(df_ml['alta_intensidade']==0).sum():,} amostras "
          f"({(df_ml['alta_intensidade']==0).sum()/len(df_ml)*100:.1f}%)")
    print(f"   • Classe 1 (Alta): {(df_ml['alta_intensidade']==1).sum():,} amostras "
          f"({(df_ml['alta_intensidade']==1).sum()/len(df_ml)*100:.1f}%)")
    
    # Selecionar features para modelagem
    features_ml = ['mes', 'dia_do_ano', 'dia_semana', 'periodo_critico', 
                   'fim_de_semana', 'densidade_diaria', 'densidade_espacial']
    
    # Adicionar coordenadas se disponíveis
    if 'latitude' in df_ml.columns and 'longitude' in df_ml.columns:
        features_ml.extend(['latitude', 'longitude'])
    
    # Adicionar features encodadas se disponíveis
    if 'estado_freq' in df_ml.columns:
        features_ml.append('estado_freq')
    if 'municipio_freq' in df_ml.columns:
        features_ml.append('municipio_freq')
    if 'intensidade_classe_encoded' in df_ml.columns:
        features_ml.append('intensidade_classe_encoded')
    
    # Verificar disponibilidade das features
    features_ml = [f for f in features_ml if f in df_ml.columns]
    
    print(f"\n📊 Features selecionadas para modelagem ({len(features_ml)}):")
    for feat in features_ml:
        print(f"   • {feat}")
    
    # Preparar X e y
    df_ml_clean = df_ml[features_ml + ['alta_intensidade']].dropna()
    X = df_ml_clean[features_ml]
    y = df_ml_clean['alta_intensidade']
    
    print(f"\n📊 Dataset para modelagem:")
    print(f"   • Total de amostras: {len(X):,}")
    print(f"   • Número de features: {X.shape[1]}")
    
    # ### 6.2 Divisão em Conjuntos de Treino e Teste
    
    # Usar 20% para teste
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"\n✂️  DIVISÃO DOS DADOS:")
    print(f"   • Treino: {len(X_train):,} amostras ({len(X_train)/len(X)*100:.1f}%)")
    print(f"   • Teste:  {len(X_test):,} amostras ({len(X_test)/len(X)*100:.1f}%)")
    
    # Distribuição das classes
    print("\n   Distribuição nos conjuntos:")
    print(f"   Treino - Classe 0: {(y_train==0).sum():,} | Classe 1: {(y_train==1).sum():,}")
    print(f"   Teste  - Classe 0: {(y_test==0).sum():,} | Classe 1: {(y_test==1).sum():,}")
    
    # ### 6.3 Treinamento do Random Forest
    
    print("\n" + "=" * 80)
    print("🌲 TREINAMENTO DO RANDOM FOREST")
    print("=" * 80)
    
    # Inicializar modelo Random Forest
    rf_model = RandomForestClassifier(
        n_estimators=100,
        max_depth=15,
        min_samples_split=10,
        min_samples_leaf=5,
        random_state=42,
        n_jobs=-1,
        verbose=1
    )
    
    print("\n⏳ Treinando Random Forest...")
    print(f"   Parâmetros: n_estimators=100, max_depth=15")
    
    rf_model.fit(X_train, y_train)
    
    print("✅ Modelo Random Forest treinado com sucesso!")
    
    # Predições
    y_pred_rf_train = rf_model.predict(X_train)
    y_pred_rf_test = rf_model.predict(X_test)
    y_pred_proba_rf = rf_model.predict_proba(X_test)[:, 1]
    
    # Métricas de desempenho
    print(f"\n📊 DESEMPENHO DO RANDOM FOREST:")
    print(f"\n   Conjunto de Treino:")
    print(f"   • Acurácia:  {accuracy_score(y_train, y_pred_rf_train):.4f}")
    print(f"   • Precisão:  {precision_score(y_train, y_pred_rf_train):.4f}")
    print(f"   • Recall:    {recall_score(y_train, y_pred_rf_train):.4f}")
    print(f"   • F1-Score:  {f1_score(y_train, y_pred_rf_train):.4f}")
    
    print(f"\n   Conjunto de Teste:")
    acc_rf = accuracy_score(y_test, y_pred_rf_test)
    prec_rf = precision_score(y_test, y_pred_rf_test)
    rec_rf = recall_score(y_test, y_pred_rf_test)
    f1_rf = f1_score(y_test, y_pred_rf_test)
    auc_rf = roc_auc_score(y_test, y_pred_proba_rf)
    
    print(f"   • Acurácia:  {acc_rf:.4f}")
    print(f"   • Precisão:  {prec_rf:.4f}")
    print(f"   • Recall:    {rec_rf:.4f}")
    print(f"   • F1-Score:  {f1_rf:.4f}")
    print(f"   • AUC-ROC:   {auc_rf:.4f}")
    
    # Matriz de Confusão
    cm_rf = confusion_matrix(y_test, y_pred_rf_test)
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Matriz de confusão
    sns.heatmap(cm_rf, annot=True, fmt='d', cmap='Blues', ax=axes[0],
                xticklabels=['Baixa/Média', 'Alta'],
                yticklabels=['Baixa/Média', 'Alta'])
    axes[0].set_title('Matriz de Confusão - Random Forest', fontsize=12, fontweight='bold')
    axes[0].set_ylabel('Valor Real')
    axes[0].set_xlabel('Valor Predito')
    
    # Curva ROC
    fpr_rf, tpr_rf, _ = roc_curve(y_test, y_pred_proba_rf)
    axes[1].plot(fpr_rf, tpr_rf, color='darkorange', lw=2, 
                 label=f'ROC curve (AUC = {auc_rf:.3f})')
    axes[1].plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Baseline')
    axes[1].set_xlim([0.0, 1.0])
    axes[1].set_ylim([0.0, 1.05])
    axes[1].set_xlabel('Taxa de Falsos Positivos')
    axes[1].set_ylabel('Taxa de Verdadeiros Positivos')
    axes[1].set_title('Curva ROC - Random Forest', fontsize=12, fontweight='bold')
    axes[1].legend(loc="lower right")
    axes[1].grid(alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # Importância das Features
    feature_importance_rf = pd.DataFrame({
        'feature': features_ml,
        'importance': rf_model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print("\n🔍 IMPORTÂNCIA DAS FEATURES (Random Forest):")
    print(feature_importance_rf.to_string(index=False))
    
    # Visualizar importância
    plt.figure(figsize=(10, 6))
    sns.barplot(data=feature_importance_rf.head(15), x='importance', y='feature', palette='viridis')
    plt.title('Top 15 Features Mais Importantes - Random Forest', fontsize=14, fontweight='bold')
    plt.xlabel('Importância')
    plt.ylabel('Feature')
    plt.tight_layout()
    plt.show()
    
    # ### 6.4 Treinamento do XGBoost
    
    print("\n" + "=" * 80)
    print("🚀 TREINAMENTO DO XGBOOST")
    print("=" * 80)
    
    # Calcular scale_pos_weight para balanceamento
    scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
    
    # Inicializar modelo XGBoost
    xgb_model = xgb.XGBClassifier(
        n_estimators=100,
        max_depth=6,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=scale_pos_weight,
        random_state=42,
        n_jobs=-1,
        eval_metric='logloss'
    )
    
    print("\n⏳ Treinando XGBoost...")
    print(f"   Parâmetros: n_estimators=100, max_depth=6, learning_rate=0.1")
    print(f"   Scale pos weight: {scale_pos_weight:.2f}")
    
    xgb_model.fit(X_train, y_train, 
                  eval_set=[(X_test, y_test)],
                  verbose=False)
    
    print("✅ Modelo XGBoost treinado com sucesso!")
    
    # Predições
    y_pred_xgb_train = xgb_model.predict(X_train)
    y_pred_xgb_test = xgb_model.predict(X_test)
    y_pred_proba_xgb = xgb_model.predict_proba(X_test)[:, 1]
    
    # Métricas de desempenho
    print(f"\n📊 DESEMPENHO DO XGBOOST:")
    print(f"\n   Conjunto de Treino:")
    print(f"   • Acurácia:  {accuracy_score(y_train, y_pred_xgb_train):.4f}")
    print(f"   • Precisão:  {precision_score(y_train, y_pred_xgb_train):.4f}")
    print(f"   • Recall:    {recall_score(y_train, y_pred_xgb_train):.4f}")
    print(f"   • F1-Score:  {f1_score(y_train, y_pred_xgb_train):.4f}")
    
    print(f"\n   Conjunto de Teste:")
    acc_xgb = accuracy_score(y_test, y_pred_xgb_test)
    prec_xgb = precision_score(y_test, y_pred_xgb_test)
    rec_xgb = recall_score(y_test, y_pred_xgb_test)
    f1_xgb = f1_score(y_test, y_pred_xgb_test)
    auc_xgb = roc_auc_score(y_test, y_pred_proba_xgb)
    
    print(f"   • Acurácia:  {acc_xgb:.4f}")
    print(f"   • Precisão:  {prec_xgb:.4f}")
    print(f"   • Recall:    {rec_xgb:.4f}")
    print(f"   • F1-Score:  {f1_xgb:.4f}")
    print(f"   • AUC-ROC:   {auc_xgb:.4f}")
    
    # Matriz de Confusão
    cm_xgb = confusion_matrix(y_test, y_pred_xgb_test)
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Matriz de confusão
    sns.heatmap(cm_xgb, annot=True, fmt='d', cmap='Greens', ax=axes[0],
                xticklabels=['Baixa/Média', 'Alta'],
                yticklabels=['Baixa/Média', 'Alta'])
    axes[0].set_title('Matriz de Confusão - XGBoost', fontsize=12, fontweight='bold')
    axes[0].set_ylabel('Valor Real')
    axes[0].set_xlabel('Valor Predito')
    
    # Curva ROC
    fpr_xgb, tpr_xgb, _ = roc_curve(y_test, y_pred_proba_xgb)
    axes[1].plot(fpr_xgb, tpr_xgb, color='green', lw=2, 
                 label=f'ROC curve (AUC = {auc_xgb:.3f})')
    axes[1].plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Baseline')
    axes[1].set_xlim([0.0, 1.0])
    axes[1].set_ylim([0.0, 1.05])
    axes[1].set_xlabel('Taxa de Falsos Positivos')
    axes[1].set_ylabel('Taxa de Verdadeiros Positivos')
    axes[1].set_title('Curva ROC - XGBoost', fontsize=12, fontweight='bold')
    axes[1].legend(loc="lower right")
    axes[1].grid(alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # Importância das Features
    feature_importance_xgb = pd.DataFrame({
        'feature': features_ml,
        'importance': xgb_model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print("\n🔍 IMPORTÂNCIA DAS FEATURES (XGBoost):")
    print(feature_importance_xgb.to_string(index=False))
    
    # Visualizar importância
    plt.figure(figsize=(10, 6))
    sns.barplot(data=feature_importance_xgb.head(15), x='importance', y='feature', palette='viridis')
    plt.title('Top 15 Features Mais Importantes - XGBoost', fontsize=14, fontweight='bold')
    plt.xlabel('Importância')
    plt.ylabel('Feature')
    plt.tight_layout()
    plt.show()
    
    # ### 6.5 Comparação de Modelos
    
    print("\n" + "=" * 80)
    print("📊 COMPARAÇÃO ENTRE MODELOS")
    print("=" * 80)
    
    # Tabela comparativa
    comparacao = pd.DataFrame({
        'Modelo': ['Random Forest', 'XGBoost'],
        'Acurácia': [acc_rf, acc_xgb],
        'Precisão': [prec_rf, prec_xgb],
        'Recall': [rec_rf, rec_xgb],
        'F1-Score': [f1_rf, f1_xgb],
        'AUC-ROC': [auc_rf, auc_xgb]
    })
    
    print("\n" + comparacao.to_string(index=False))
    
    # Visualização comparativa
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=('Comparação de Métricas', 'Curvas ROC'),
        specs=[[{'type': 'bar'}, {'type': 'scatter'}]]
    )
    
    # Gráfico de barras comparativo
    metricas = ['Acurácia', 'Precisão', 'Recall', 'F1-Score', 'AUC-ROC']
    for idx, modelo in enumerate(['Random Forest', 'XGBoost']):
        valores = comparacao[comparacao['Modelo'] == modelo][metricas].values[0]
        fig.add_trace(
            go.Bar(x=metricas, y=valores, name=modelo,
                   marker_color='darkorange' if idx == 0 else 'green'),
            row=1, col=1
        )
    
    # Curvas ROC sobrepostas
    fig.add_trace(
        go.Scatter(x=fpr_rf, y=tpr_rf, mode='lines', name=f'Random Forest (AUC={auc_rf:.3f})',
                   line=dict(color='darkorange', width=2)),
        row=1, col=2
    )
    fig.add_trace(
        go.Scatter(x=fpr_xgb, y=tpr_xgb, mode='lines', name=f'XGBoost (AUC={auc_xgb:.3f})',
                   line=dict(color='green', width=2)),
        row=1, col=2
    )
    fig.add_trace(
        go.Scatter(x=[0, 1], y=[0, 1], mode='lines', name='Baseline',
                   line=dict(color='navy', width=2, dash='dash')),
        row=1, col=2
    )
    
    fig.update_xaxes(title_text="Métricas", row=1, col=1)
    fig.update_yaxes(title_text="Valor", row=1, col=1)
    fig.update_xaxes(title_text="Taxa de Falsos Positivos", row=1, col=2)
    fig.update_yaxes(title_text="Taxa de Verdadeiros Positivos", row=1, col=2)
    
    fig.update_layout(height=500, title_text="🏆 Comparação de Desempenho dos Modelos Preditivos",
                      showlegend=True, barmode='group')
    fig.show()
    
    # Determinar melhor modelo
    melhor_modelo_nome = comparacao.loc[comparacao['F1-Score'].idxmax(), 'Modelo']
    melhor_f1 = comparacao['F1-Score'].max()
    
    print(f"\n🏆 MELHOR MODELO: {melhor_modelo_nome}")
    print(f"   F1-Score: {melhor_f1:.4f}")
    
    # ### 6.6 Validação Cruzada
    
    print("\n" + "=" * 80)
    print("🔄 VALIDAÇÃO CRUZADA (K-Fold)")
    print("=" * 80)
    
    # Random Forest
    print("\n⏳ Executando validação cruzada para Random Forest...")
    cv_scores_rf = cross_val_score(rf_model, X, y, cv=5, scoring='f1', n_jobs=-1)
    print(f"   F1-Scores por fold: {cv_scores_rf}")
    print(f"   Média: {cv_scores_rf.mean():.4f} (+/- {cv_scores_rf.std():.4f})")
    
    # XGBoost
    print("\n⏳ Executando validação cruzada para XGBoost...")
    cv_scores_xgb = cross_val_score(xgb_model, X, y, cv=5, scoring='f1', n_jobs=-1)
    print(f"   F1-Scores por fold: {cv_scores_xgb}")
    print(f"   Média: {cv_scores_xgb.mean():.4f} (+/- {cv_scores_xgb.std():.4f})")
    
    # Visualização dos resultados de CV
    cv_results = pd.DataFrame({
        'Fold': list(range(1, 6)) * 2,
        'F1-Score': list(cv_scores_rf) + list(cv_scores_xgb),
        'Modelo': ['Random Forest'] * 5 + ['XGBoost'] * 5
    })
    
    fig = px.box(cv_results, x='Modelo', y='F1-Score', color='Modelo',
                 title='Distribuição dos F1-Scores na Validação Cruzada (5-Fold)',
                 color_discrete_map={'Random Forest': 'darkorange', 'XGBoost': 'green'})
    fig.update_layout(height=500)
    fig.show()
    
    print("\n✅ Modelagem Preditiva concluída!")

else:
    print("\n⚠️  Coluna de FRP não encontrada. Pulando modelagem preditiva supervisionada.")

# ---

# ## 7️⃣ Etapa 6: Análise de Predição Espacial e Temporal

print("\n" + "=" * 80)
print("🗺️ ANÁLISE DE PREDIÇÃO ESPACIAL E TEMPORAL")
print("=" * 80)

if col_frp in df_processed.columns and 'alta_intensidade' in df_ml.columns:
    
    # Adicionar predições ao dataframe original
    df_ml_clean['pred_rf'] = y_pred_rf_test
    df_ml_clean['pred_xgb'] = y_pred_xgb_test
    df_ml_clean['proba_rf'] = y_pred_proba_rf
    df_ml_clean['proba_xgb'] = y_pred_proba_xgb
    
    # Análise de erros por região
    if 'estado' in df_ml_clean.columns:
        print("\n📊 ANÁLISE DE ERROS POR ESTADO:")
        
        # Calcular métricas por estado para o melhor modelo
        melhor_modelo_pred = 'pred_rf' if melhor_modelo_nome == 'Random Forest' else 'pred_xgb'
        
        estados_metricas = []
        for estado in df_ml_clean['estado'].unique()[:10]:  # Top 10 estados
            df_estado = df_ml_clean[df_ml_clean['estado'] == estado]
            if len(df_estado) > 50:  # Mínimo de amostras
                acc_estado = accuracy_score(df_estado['alta_intensidade'], 
                                            df_estado[melhor_modelo_pred])
                estados_metricas.append({
                    'Estado': estado,
                    'Amostras': len(df_estado),
                    'Acurácia': acc_estado
                })
        
        df_estados_metricas = pd.DataFrame(estados_metricas).sort_values('Acurácia', ascending=False)
        print(df_estados_metricas.to_string(index=False))
    
    # Mapa de predições
    if 'latitude' in df_ml_clean.columns and 'longitude' in df_ml_clean.columns:
        print("\n🗺️  Gerando mapa de predições...")
        
        # Amostrar para visualização
        df_map_sample = df_ml_clean.sample(min(5000, len(df_ml_clean)))
        
        # Recuperar coordenadas originais
        lat_map = df_processed.loc[df_map_sample.index, 'latitude']
        lon_map = df_processed.loc[df_map_sample.index, 'longitude']
        
        # Classificação: Correto vs Incorreto
        melhor_pred = 'pred_rf' if melhor_modelo_nome == 'Random Forest' else 'pred_xgb'
        df_map_sample['classificacao'] = df_map_sample.apply(
            lambda row: 'Correto' if row['alta_intensidade'] == row[melhor_pred] else 'Incorreto',
            axis=1
        )
        
        fig = px.scatter_mapbox(
            df_map_sample,
            lat=lat_map,
            lon=lon_map,
            color='classificacao',
            color_discrete_map={'Correto': 'green', 'Incorreto': 'red'},
            zoom=4,
            height=600,
            title=f'🗺️ Mapa de Predições - {melhor_modelo_nome} (Amostra)',
            labels={'classificacao': 'Classificação'},
            hover_data=['alta_intensidade', melhor_pred]
        )
        
        fig.update_layout(mapbox_style="open-street-map")
        fig.update_traces(marker=dict(size=5, opacity=0.6))
        fig.show()

# ---

# ## 8️⃣ Etapa 7: Insights e Recomendações

print("\n" + "=" * 80)
print("💡 INSIGHTS E RECOMENDAÇÕES ESTRATÉGICAS")
print("=" * 80)

print("\n" + "🔥" * 40)
print("PRINCIPAIS INSIGHTS EXTRAÍDOS DA ANÁLISE")
print("🔥" * 40)

# Insight 1: Padrões Temporais
print("\n1️⃣  PADRÕES TEMPORAIS:")
if 'mes' in df.columns:
    mes_critico = ocorrencias_mensais.loc[ocorrencias_mensais['ocorrencias'].idxmax(), 'nome_mes']
    focos_mes_critico = ocorrencias_mensais['ocorrencias'].max()
    print(f"   • Mês mais crítico: {mes_critico} com {focos_mes_critico:,} focos")
    print(f"   • Período de seca (jul-out) concentra {df[df['periodo_critico']==1].shape[0]/len(df)*100:.1f}% dos focos")
    print(f"   • Recomendação: Intensificar monitoramento e recursos nos meses de julho a outubro")

# Insight 2: Distribuição Espacial
print("\n2️⃣  DISTRIBUIÇÃO ESPACIAL:")
if 'estado' in df.columns:
    estado_critico = ocorrencias_estado.iloc[0]['estado']
    focos_estado = ocorrencias_estado.iloc[0]['ocorrencias']
    print(f"   • Estado mais afetado: {estado_critico} ({focos_estado:,} focos)")
    print(f"   • Top 3 estados concentram {ocorrencias_estado.head(3)['ocorrencias'].sum()/len(df)*100:.1f}% dos focos")
    print(f"   • Recomendação: Criar centros regionais de resposta rápida nos estados mais afetados")

# Insight 3: Clusters Identificados
print("\n3️⃣  PADRÕES DE AGRUPAMENTO:")
if 'cluster' in df_processed.columns:
    n_clusters_final = df_processed['cluster'].nunique()
    print(f"   • {n_clusters_final} clusters distintos identificados com características únicas")
    print(f"   • Clusters revelam padrões espaciais, temporais e de intensidade")
    print(f"   • Recomendação: Desenvolver estratégias específicas para cada tipo de cluster")

# Insight 4: Variáveis Preditivas
print("\n4️⃣  FATORES PREDITIVOS:")
if 'alta_intensidade' in locals() and melhor_modelo_nome:
    top_features = feature_importance_rf if melhor_modelo_nome == 'Random Forest' else feature_importance_xgb
    print(f"   • Features mais importantes para predição:")
    for idx, row in top_features.head(5).iterrows():
        print(f"      {idx+1}. {row['feature']}: {row['importance']:.4f}")
    print(f"   • Modelo {melhor_modelo_nome} atingiu F1-Score de {melhor_f1:.4f}")
    print(f"   • Recomendação: Monitorar ativamente as variáveis de maior importância preditiva")

# Insight 5: Intensidade dos Focos
print("\n5️⃣  INTENSIDADE DAS QUEIMADAS:")
if col_frp in df.columns:
    frp_medio = df[col_frp].mean()
    frp_max = df[col_frp].max()
    focos_alta_intensidade = (df[col_frp] > limiar_intensidade).sum()
    print(f"   • Intensidade média (FRP): {frp_medio:.2f} MW")
    print(f"   • Intensidade máxima registrada: {frp_max:.2f} MW")
    print(f"   • {focos_alta_intensidade:,} focos classificados como alta intensidade")
    print(f"   • Recomendação: Priorizar combate a focos com FRP > {limiar_intensidade:.1f} MW")

print("\n" + "=" * 80)
print("🎯 RECOMENDAÇÕES ESTRATÉGICAS PARA GESTÃO E PREVENÇÃO")
print("=" * 80)

recomendacoes = [
    {
        'titulo': '🚨 Sistema de Alerta Precoce',
        'descricao': 'Implementar sistema automatizado de alertas baseado nos modelos preditivos desenvolvidos, '
                     'com limiares de probabilidade calibrados para diferentes níveis de ação (amarelo, laranja, vermelho).',
        'prazo': 'Curto prazo (3-6 meses)',
        'impacto': 'Alto'
    },
    {
        'titulo': '🗺️ Mapeamento de Áreas Prioritárias',
        'descricao': 'Criar mapas de risco atualizados mensalmente identificando hotspots críticos para '
                     'alocação otimizada de equipes de monitoramento e combate.',
        'prazo': 'Imediato (1-3 meses)',
        'impacto': 'Alto'
    },
    {
        'titulo': '📊 Dashboard de Monitoramento em Tempo Real',
        'descricao': 'Desenvolver plataforma web interativa consolidando dados de satélite, previsões dos modelos, '
                     'alertas ativos e status de recursos de combate, acessível para gestores e brigadas.',
        'prazo': 'Médio prazo (6-12 meses)',
        'impacto': 'Muito Alto'
    },
    {
        'titulo': '👥 Capacitação de Equipes Locais',
        'descricao': 'Treinar comunidades locais, fazendeiros e brigadistas em identificação precoce de focos, '
                     'uso de aplicativos de reporte, e técnicas de prevenção baseadas nos padrões identificados.',
        'prazo': 'Contínuo',
        'impacto': 'Médio-Alto'
    },
    {
        'titulo': '🌱 Políticas de Uso Sustentável do Solo',
        'descricao': 'Estabelecer regulamentações mais rigorosas para queimadas controladas nos períodos críticos, '
                     'incentivando práticas agrícolas alternativas ao uso do fogo.',
        'prazo': 'Longo prazo (1-2 anos)',
        'impacto': 'Alto'
    },
    {
        'titulo': '🛰️ Integração com Dados Meteorológicos',
        'descricao': 'Incorporar variáveis meteorológicas de alta resolução (temperatura, umidade, vento) aos '
                     'modelos preditivos para aumentar acurácia e antecipação de eventos críticos.',
        'prazo': 'Médio prazo (6-12 meses)',
        'impacto': 'Alto'
    },
    {
        'titulo': '🤝 Parcerias Interinstitucionais',
        'descricao': 'Estabelecer cooperação entre INPE, IBAMA, ICMBio, Corpo de Bombeiros, universidades e ONGs '
                     'para compartilhamento de dados, recursos e expertise técnica.',
        'prazo': 'Curto prazo (3-6 meses)',
        'impacto': 'Médio'
    }
]

for idx, rec in enumerate(recomendacoes, 1):
    print(f"\n{idx}. {rec['titulo']}")
    print(f"   📋 Descrição: {rec['descricao']}")
    print(f"   ⏰ Prazo: {rec['prazo']}")
    print(f"   💥 Impacto Esperado: {rec['impacto']}")

# ---

# ## 9️⃣ Conclusões e Trabalhos Futuros

print("\n" + "=" * 80)
print("📝 CONCLUSÕES E TRABALHOS FUTUROS")
print("=" * 80)

print("\n" + "✅" * 40)
print("CONCLUSÕES PRINCIPAIS")
print("✅" * 40)

print("""
Este projeto desenvolveu um sistema inteligente abrangente para análise, monitoramento e predição 
de queimadas no Pantanal utilizando técnicas avançadas de Ciência de Dados e Aprendizado de Máquina 
aplicadas a dados geoespaciais reais de 2020.

🎯 PRINCIPAIS CONQUISTAS:

1. Análise Exploratória Robusta:
   • Processamento e análise de milhares de registros de focos de queimadas
   • Identificação de padrões temporais, espaciais e de intensidade
   • Caracterização detalhada da crise de queimadas de 2020 no Pantanal

2. Modelagem Não Supervisionada:
   • Aplicação bem-sucedida de K-Means e DBSCAN para identificação de clusters
   • Descoberta de agrupamentos naturais com características distintas
   • Validação através de múltiplas métricas (silhueta, Davies-Bouldin, Calinski-Harabasz)

3. Modelagem Preditiva:
   • Desenvolvimento de modelos Random Forest e XGBoost com alta acurácia
   • Identificação das variáveis mais importantes para predição
   • Validação rigorosa através de validação cruzada e métricas múltiplas

4. Visualizações Informativas:
   • Mapas interativos mostrando distribuição espacial dos focos
   • Séries temporais revelando sazonalidade e tendências
   • Dashboards integrados facilitando interpretação dos resultados

5. Insights Acionáveis:
   • Identificação de períodos e regiões críticas para intensificação do monitoramento
   • Recomendações concretas para políticas públicas e estratégias de prevenção
   • Base científica sólida para tomada de decisão em gestão ambiental

📊 CONTRIBUIÇÕES CIENTÍFICAS:

• Demonstração da aplicabilidade de técnicas de ML em problemas ambientais complexos
• Metodologia replicável para análise de queimadas em outros biomas brasileiros
• Integração efetiva entre análise exploratória, modelagem e visualização geoespacial
• Framework para desenvolvimento de sistemas operacionais de alerta precoce

⚠️ LIMITAÇÕES IDENTIFICADAS:

• Dados limitados a um único ano (2020) - necessidade de séries históricas mais longas
• Ausência de variáveis meteorológicas detalhadas que poderiam melhorar predições
• Possíveis vieses de detecção por satélite (cobertura de nuvens, resolução temporal)
• Modelos não consideram explicitamente autocorrelação espacial dos dados

🔮 TRABALHOS FUTUROS RECOMENDADOS:

1. Expansão Temporal:
   • Incorporar dados de múltiplos anos (2015-2024) para análise de tendências de longo prazo
   • Desenvolver modelos de séries temporais (ARIMA, LSTM) para predição sequencial

2. Enriquecimento de Dados:
   • Integrar variáveis meteorológicas (temperatura, umidade, precipitação, vento)
   • Adicionar dados de uso do solo, proximidade a áreas urbanas e estradas
   • Incorporar índices de vegetação derivados de imagens de satélite (NDVI, EVI)

3. Modelagem Avançada:
   • Implementar modelos espacialmente explícitos (Spatial Random Forest, GWR)
   • Desenvolver redes neurais deep learning (CNN para imagens, LSTM para séries temporais)
   • Explorar ensemble methods combinando múltiplos algoritmos

4. Sistema Operacional:
   • Desenvolver API para consumo de predições em tempo real
   • Criar aplicativo móvel para brigadistas e comunidades locais
   • Implementar pipeline automatizado de atualização de dados e retreinamento de modelos

5. Análise de Impactos:
   • Avaliar danos ambientais e socioeconômicos associados às queimadas
   • Estimar emissões de carbono e impactos climáticos
   • Analisar efeitos na biodiversidade e em espécies ameaçadas

6. Validação de Campo:
   • Realizar validações in loco das predições dos modelos
   • Coletar dados de campo para calibração e melhoria dos algoritmos
   • Estabelecer parceria com brigadas para feedback sobre utilidade operacional

""")

print("=" * 80)
print("🏆 PROJETO CONCLUÍDO COM SUCESSO!")
print("=" * 80)
print(f"""
📊 Estatísticas Finais do Projeto:
   • Dataset analisado: {len(df):,} registros de focos de queimadas
   • Período: {df['data'].min()} a {df['data'].max() if 'data' in df.columns else 'N/A'}
   • Features criadas: {len([c for c in df_processed.columns if c not in df_original.columns])}
   • Modelos desenvolvidos: 2+ (K-Means, Random Forest, XGBoost)
   • Visualizações geradas: 15+
   • Insights extraídos: 5 principais
   • Recomendações estratégicas: 7

🌿 Contribuição para Conservação do Pantanal:
Este projeto fornece ferramentas científicas robustas e acionáveis para apoiar a 
conservação de um dos biomas mais ricos e ameaçados do planeta, contribuindo para
a proteção da biodiversidade, sustentabilidade ambiental e bem-estar das comunidades
que dependem deste ecossistema único.

✨ Ciência de Dados a Serviço da Preservação Ambiental ✨
""")

print("\n🎓 Projeto desenvolvido para a disciplina de Aplicações em Aprendizado de Máquina")
print("🏫 Curso: Ciência de Dados")
print("📅 Ano: 2025")
print("\n" + "🔥" * 40)
print("OBRIGADO!")
print("🔥" * 40)

# FIM DO NOTEBOOK 
