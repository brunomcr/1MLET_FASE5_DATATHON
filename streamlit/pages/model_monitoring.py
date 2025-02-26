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

def calculate_f1_score(precision, recall):
    """Calcula o F1 Score"""
    if precision + recall == 0:
        return 0.0
    return 2 * (precision * recall) / (precision + recall)

def calculate_gini_index(interaction_data):
    """Calcula o índice de Gini para medir a diversidade das recomendações"""
    total_interactions = interaction_data['total_interactions']
    if total_interactions == 0:
        return 0.0
    
    # Usando interações por item como medida de distribuição
    interactions_per_item = interaction_data['interactions_per_item']
    return 1 - (interactions_per_item * 2)  # Normalizado entre 0 e 1

def calculate_conversion_rate(interaction_data):
    """Calcula a taxa de conversão das recomendações"""
    if interaction_data['train_interactions'] == 0:
        return 0.0
    return interaction_data['test_interactions'] / interaction_data['train_interactions']

def plot_distribution_metrics(interaction_data):
    """Plota métricas de distribuição"""
    fig = go.Figure()
    
    # Criar dados para o gráfico
    categories = ['Interações por Usuário', 'Interações por Item']
    values = [interaction_data['interactions_per_user'], 
              interaction_data['interactions_per_item']]
    
    # Adicionar barras
    fig.add_trace(go.Bar(
        x=categories,
        y=values,
        name='Média de Interações'
    ))
    
    fig.update_layout(
        title='Distribuição Média de Interações',
        yaxis_title='Média de Interações',
        showlegend=False
    )
    
    return fig

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
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📋 Resumo do Modelo",
        "📈 Performance",
        "📊 Distribuição",
        "🎯 Métricas Avançadas",
        "⚙️ Métricas Técnicas"
    ])
    
    with tab1:
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Informações do Modelo")
            model_summary = results["model_summary"]
            st.markdown(f"""
            - **Tipo**: {model_summary['model_type']}
            - **Dimensão dos Embeddings**: {model_summary['embedding_dim']}
            - **Função de Perda**: {model_summary['loss_function']}
            - **Learning Rate**: {model_summary['learning_rate']}
            - **Tamanho do Modelo**: {model_summary['model_size_mb']:.2f} MB
            ------------------------------------------------------------
            """)
            st.markdown("""
            **Explicação das Métricas:**
            - **Tipo**: Tipo de modelo utilizado, como LightFM, que é adequado para recomendações.
            - **Dimensão dos Embeddings**: Tamanho dos vetores que representam usuários e itens, influenciando a capacidade do modelo de capturar nuances.
            - **Função de Perda**: Método usado para ajustar o modelo, determinando como os erros são penalizados.
            - **Learning Rate**: Taxa de aprendizado que controla a velocidade de ajuste do modelo durante o treinamento.
            - **Tamanho do Modelo**: Espaço ocupado pelo modelo em disco, importante para armazenamento e carregamento.
            """)
        
        with col2:
            st.subheader("Hiperparâmetros")
            hyperparams = results["hyperparameters"]
            st.markdown(f"""
            - **Número de Componentes**: {hyperparams['no_components']}
            - **Learning Rate**: {hyperparams['learning_rate']}
            - **Loss Function**: {hyperparams['loss']}
            - **Item Alpha**: {hyperparams['item_alpha']}
            - **User Alpha**: {hyperparams['user_alpha']}
            - **Random State**: {hyperparams['random_state']}
            ------------------------------------------------------------
            """)
            st.markdown("""
            **Explicação dos Hiperparâmetros:**
            - **Número de Componentes**: Quantidade de características latentes que o modelo aprende, afetando a precisão.
            - **Learning Rate**: Controla a rapidez com que o modelo se ajusta aos dados, impactando a convergência.
            - **Loss Function**: Mede o erro do modelo, influenciando a qualidade das previsões.
            - **Item Alpha**: Regularização para itens, ajuda a evitar overfitting.
            - **User Alpha**: Regularização para usuários, também ajuda a evitar overfitting.
            - **Random State**: Garante a reprodutibilidade dos resultados ao fixar a semente aleatória.
            """)

        # Adicionar métricas de convergência se disponíveis
        if 'convergence_metrics' in results:
            st.subheader("Métricas de Convergência")
            conv_metrics = results['convergence_metrics']
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Loss Final", f"{conv_metrics.get('final_loss', 'N/A')}")
                st.metric("Épocas até Convergência", f"{conv_metrics.get('epochs_until_convergence', 'N/A')}")
            with col2:
                st.metric("Época do Early Stopping", f"{conv_metrics.get('early_stopping_epoch', 'N/A')}")
    
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
            
        # Calcular e exibir o F1 Score
        f1_score = calculate_f1_score(metrics['precision@10'], metrics['recall@10'])
        st.metric("F1 Score", f"{f1_score:.4f}")
        
        st.markdown("""
        **Explicação das Métricas de Performance:**
        - **Precision@10**: Mede a proporção de itens relevantes entre os 10 primeiros recomendados, indicando precisão.
        - **Recall@10**: Mede a proporção de itens relevantes recomendados entre os 10 primeiros, indicando cobertura.
        - **AUC**: Área sob a curva ROC, avalia a capacidade do modelo de distinguir entre classes.
        - **F1 Score**: Combina precisão e recall em uma única métrica, útil para balancear ambos.
        ------------------------------------------------------------
        """)
        
        st.subheader("Métricas de Estabilidade")
        stability = results["model_stability"]
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Cobertura de Usuários", f"{stability['user_coverage']:.2%}")
            st.metric("Usuários Cold-Start", f"{stability['cold_start_users']:,}")
        with col2:
            st.metric("Cobertura de Items", f"{stability['item_coverage']:.2%}")
            st.metric("Items Cold-Start", f"{stability['cold_start_items']:,}")
        
        st.markdown("""
        **Explicação das Métricas de Estabilidade:**
        - **Cobertura de Usuários**: Proporção de usuários que receberam recomendações, importante para alcance.
        - **Usuários Cold-Start**: Usuários sem histórico de interações, desafiadores para recomendações.
        - **Cobertura de Items**: Proporção de itens recomendados, importante para diversidade.
        - **Items Cold-Start**: Itens sem histórico de interações, desafiadores para recomendações.
        ------------------------------------------------------------
        """)
    
    with tab3:
        st.subheader("Distribuição de Interações")
        fig_interactions = plot_interaction_metrics(results["interaction_distribution"])
        st.plotly_chart(fig_interactions, use_container_width=True)
        
        # Adicionar gráfico de distribuição média
        st.markdown("""------------------------------------------------------------""")
        fig_distribution = plot_distribution_metrics(results["interaction_distribution"])
        st.plotly_chart(fig_distribution, use_container_width=True)
        
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
        
        st.markdown("""
        **Explicação da Distribuição de Interações:**
        - **Interações por Usuário**: Média de interações por usuário, reflete engajamento.
        - **Interações por Item**: Média de interações por item, reflete popularidade.
        - **Sparsidade da Matriz**: Proporção de elementos vazios, indica densidade de dados.
        ------------------------------------------------------------
        """)
        
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
        
        st.markdown("""
        **Explicação das Informações do Dataset:**
        - **Número de Usuários**: Total de usuários no dataset, importante para escala.
        - **Número de Items**: Total de itens no dataset, importante para variedade.
        - **Número de Features**: Total de características usadas no modelo, influencia a complexidade.
        ------------------------------------------------------------
        """)

    with tab4:
        st.subheader("Métricas Avançadas")
        
        # Calcular novas métricas
        user_coverage = calculate_user_coverage(results['interaction_distribution'], dataset_info['n_users'])
        item_coverage = calculate_item_coverage(results['interaction_distribution'], dataset_info['n_items'])
        hit_rate = calculate_hit_rate(results['interaction_distribution'], dataset_info['n_users'])
        gini_index = calculate_gini_index(results['interaction_distribution'])
        conversion_rate = calculate_conversion_rate(results['interaction_distribution'])

        col1, col2 = st.columns(2)
        with col1:
            st.metric("Cobertura de Usuários", f"{user_coverage:.2%}")
            st.metric("Cobertura de Itens", f"{item_coverage:.2%}")
            st.metric("Hit Rate", f"{hit_rate:.2%}")
        
        with col2:
            st.metric("Índice de Diversidade (Gini)", f"{gini_index:.4f}")
            st.metric("Taxa de Conversão", f"{conversion_rate:.2%}")
            
        st.markdown("""
        **Explicação das Métricas:**
        - **Cobertura de Usuários**: Proporção de usuários que receberam recomendações
        - **Cobertura de Itens**: Proporção de itens que foram recomendados
        - **Hit Rate**: Taxa de acerto das recomendações
        - **Índice de Diversidade**: Medida de quão diversas são as recomendações (0-1, quanto maior, mais diverso)
        - **Taxa de Conversão**: Proporção de recomendações que resultaram em interações
        ------------------------------------------------------------
        """)

    with tab5:
        st.subheader("Métricas Técnicas")
        
        # Estatísticas dos Embeddings
        st.subheader("Estatísticas dos Embeddings")
        if 'embedding_stats' in results:
            emb_stats = results['embedding_stats']
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Média Norm. Emb. Usuários", 
                         f"{emb_stats.get('user_embedding_norm_mean', 'N/A'):.4f}")
                st.metric("Desvio Norm. Emb. Usuários", 
                         f"{emb_stats.get('user_embedding_norm_std', 'N/A'):.4f}")
            with col2:
                st.metric("Média Norm. Emb. Items", 
                         f"{emb_stats.get('item_embedding_norm_mean', 'N/A'):.4f}")
                st.metric("Desvio Norm. Emb. Items", 
                         f"{emb_stats.get('item_embedding_norm_std', 'N/A'):.4f}")
        else:
            st.info("Estatísticas dos embeddings não disponíveis")

        st.markdown("""
        **Explicação das Métricas Técnicas:**
        
        **Estatísticas dos Embeddings:**
        - Métricas que mostram a distribuição dos embeddings de usuários e items
        - Valores muito altos ou baixos podem indicar problemas no treinamento
        ------------------------------------------------------------
        """)

if __name__ == "__main__":
    show_monitoring_page() 