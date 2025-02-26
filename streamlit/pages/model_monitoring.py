import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import json
import os

def load_monitoring_results():
    """Carrega os resultados mais recentes do monitoramento"""
    try:
        current_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(os.path.dirname(current_dir))
        monitoring_path = os.path.join(project_root, 'models', 'monitoring')
        
        if not os.path.exists(monitoring_path):
            return None
        
        monitoring_files = [f for f in os.listdir(monitoring_path) 
                          if f.startswith('monitoring_results_') and f.endswith('.json')]
        
        if not monitoring_files:
            return None
        
        latest_file = max(monitoring_files)
        file_path = os.path.join(monitoring_path, latest_file)
        
        with open(file_path, 'r') as f:
            return json.load(f)
            
    except Exception as e:
        return None

def plot_feature_importance(feature_data):
    """Plota o gráfico de importância das features"""
    # Criar DataFrame com as top 20 features
    df = pd.DataFrame({
        'Feature': [f"Feature {i}" for i in range(len(feature_data))],
        'Importance': list(feature_data.values())
    }).sort_values('Importance', ascending=False).head(20)

    fig = px.bar(df, 
                 x='Feature', 
                 y='Importance',
                 title='Top 20 Features Mais Importantes',
                 color='Importance',
                 color_continuous_scale='viridis')
    
    fig.update_layout(
        xaxis_title="Features",
        yaxis_title="Importância Relativa",
        showlegend=False
    )
    
    return fig

def plot_interaction_metrics(interaction_data):
    """Plota métricas de interação"""
    fig = go.Figure()
    
    # Adiciona barras para diferentes tipos de interações
    fig.add_trace(go.Bar(
        x=['Total', 'Treino', 'Teste'],
        y=[interaction_data['total_interactions'],
           interaction_data['train_interactions'],
           interaction_data['test_interactions']],
        name='Número de Interações'
    ))
    
    fig.update_layout(
        title='Distribuição de Interações',
        yaxis_title='Número de Interações',
        showlegend=True
    )
    
    return fig

def calculate_user_coverage(interaction_data, n_users):
    """Calcula a cobertura de usuários"""
    users_with_recommendations = interaction_data['train_interactions'] + interaction_data['test_interactions']
    return users_with_recommendations / n_users


def calculate_item_coverage(interaction_data, n_items):
    """Calcula a cobertura de itens"""
    items_with_recommendations = interaction_data['train_interactions'] + interaction_data['test_interactions']
    return items_with_recommendations / n_items


def calculate_hit_rate(interaction_data, n_users):
    """Calcula o Hit Rate"""
    # Supondo que cada interação no conjunto de teste é um hit
    hits = interaction_data['test_interactions']
    return hits / n_users

def show_monitoring_page():
    st.markdown("<h1 style='font-size: 32px;'>Monitoramento do Modelo de Recomendação</h1>", unsafe_allow_html=True)
    
    # Carregar dados
    results = load_monitoring_results()
    
    if results is None:
        st.error("Não foi possível carregar os dados de monitoramento.")
        return
    
    # Timestamp do monitoramento
    st.info(f"📅 Última Atualização: {results['timestamp']}")
    
    # Layout em tabs para melhor organização
    tab1, tab2, tab3 = st.tabs([
        "📋 Resumo do Modelo",
        "📈 Performance",
        "📊 Distribuição"
    ])
    
    with tab1:
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Informações do Modelo")
            model_summary = results["model_summary"]
            st.info(f"""
            - **Tipo**: {model_summary['model_type']}
            - **Dimensão dos Embeddings**: {model_summary['embedding_dim']}
            - **Função de Perda**: {model_summary['loss_function']}
            - **Learning Rate**: {model_summary['learning_rate']}
            - **Tamanho do Modelo**: {model_summary['model_size_mb']:.2f} MB
            """)
        
        with col2:
            st.subheader("Hiperparâmetros")
            hyperparams = results["hyperparameters"]
            st.info(f"""
            - **Número de Componentes**: {hyperparams['no_components']}
            - **Learning Rate**: {hyperparams['learning_rate']}
            - **Loss Function**: {hyperparams['loss']}
            - **Item Alpha**: {hyperparams['item_alpha']}
            - **User Alpha**: {hyperparams['user_alpha']}
            - **Random State**: {hyperparams['random_state']}
            """)
    
    with tab2:
        st.subheader("Métricas de Performance")
        metrics = results["performance_metrics"]
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Precision@10", f"{metrics['precision@10']:.4f}")
        with col2:
            st.metric("Recall@10", f"{metrics['recall@10']:.4f}")
        with col3:
            st.metric("AUC", f"{metrics['auc']:.4f}")
            
        st.subheader("Métricas de Estabilidade")
        stability = results["model_stability"]
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Cobertura de Usuários", f"{stability['user_coverage']:.2%}")
            st.metric("Usuários Cold-Start", f"{stability['cold_start_users']:,}")
        with col2:
            st.metric("Cobertura de Items", f"{stability['item_coverage']:.2%}")
            st.metric("Items Cold-Start", f"{stability['cold_start_items']:,}")
    
    with tab3:
        st.subheader("Distribuição de Interações")
        fig_interactions = plot_interaction_metrics(results["interaction_distribution"])
        st.plotly_chart(fig_interactions, use_container_width=True)
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric(
                "Interações por Usuário",
                f"{results['interaction_distribution']['interactions_per_user']:.4f}"
            )
        with col2:
            st.metric(
                "Interações por Item",
                f"{results['interaction_distribution']['interactions_per_item']:.4f}"
            )
        
        st.metric(
            "Sparsidade da Matriz",
            f"{results['interaction_distribution']['sparsity']:.4%}"
        )
        
        # Informações do Dataset
        st.subheader("Informações do Dataset")
        dataset_info = results["dataset_info"]
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Número de Usuários", f"{dataset_info['n_users']:,}")
        with col2:
            st.metric("Número de Items", f"{dataset_info['n_items']:,}")
        with col3:
            st.metric("Número de Features", f"{dataset_info['n_features']:,}")

        # Calcular e exibir novas métricas
        user_coverage = calculate_user_coverage(results['interaction_distribution'], dataset_info['n_users'])
        item_coverage = calculate_item_coverage(results['interaction_distribution'], dataset_info['n_items'])
        hit_rate = calculate_hit_rate(results['interaction_distribution'], dataset_info['n_users'])

        st.subheader("Novas Métricas")
        st.metric("Cobertura de Usuários", f"{user_coverage:.2%}")
        st.metric("Cobertura de Itens", f"{item_coverage:.2%}")
        st.metric("Hit Rate", f"{hit_rate:.2%}")

if __name__ == "__main__":
    show_monitoring_page() 